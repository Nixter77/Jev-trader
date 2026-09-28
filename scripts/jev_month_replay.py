#!/usr/bin/env python3
"""Rebuild the bot's Jev requests for a historical range and estimate calls, tokens and cost.

DRY RUN IS THE DEFAULT: no model call is made, no API key is read. The live path
exists for a later, approved run only; it needs BOTH `--live` and
`--i-have-approval` plus --max-calls, reads TYPESAFE_API_KEY from the process
environment at run time (never from a file, never logged) and:
  - asks one FLAT buy question per planned bar, bar-major (all symbols advance
    together), with --min-interval seconds between calls;
  - appends every answer (raw judgment + SDK usage input/output tokens) to an
    append-only JSONL; a rerun skips bars already there (resume, no double pay);
  - budget guard from real usage at --price-per-mtok: after --check-after calls
    (2000) it projects cost/call x planned calls and stops if that exceeds
    --budget-usd; it also stops before a call that would pass the budget, at
    --max-calls (all answers in the file count) and if usage is missing;
  - retries transport errors (3 attempts, backoff), never retries an answer
    that was billed but unparsable, stops on auth/permission/bad-request errors
    and after 5 bars in a row that failed.

States are built with the bot's own code, so prompts match production:
  candles  public 5m klines of the candle venue (production reads the public
           mainnet host fapi.binance.com for features, also when it trades on
           testnet), last --window closed bars (the live CandleBuffer holds up
           to 300; it starts at 239 after a restart)
  features jev_trader.features.compute_features
  state    jev_trader.live.snapshot_from_closed_bars + jev_trader.state.build_compact_state
  request  jev_trader.jev.build_system_one_payload with the FLAT allowed actions
           (buy_long / hold), serialized like typesafe_sdk's system_one body
Flat-book assumptions: position FLAT, size 0, cash_usdt = --equity for every
call; no order book (book_imb / spread print as n/a, production had live depth-5
values); funding / doi_1h / btc_corr / news stay empty exactly as in production.

Token counts: a real tokenizer if one is available (--tokenizer, default: the
local Laya tokenizer.json from the HF cache; it is NOT Jev's tokenizer, so it is
a proxy), plus chars/4 and chars/3.5 estimates. The billable count is decided by
the Jev server, which wraps state and questions in its own prompt; that overhead
is unknown here.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from jev_trader.features import compute_features  # noqa: E402
from jev_trader.jev import allowed_actions_for_position, build_system_one_payload  # noqa: E402
from jev_trader.live import snapshot_from_closed_bars  # noqa: E402
from jev_trader.models import Candle, Position  # noqa: E402
from jev_trader.state import build_compact_state  # noqa: E402

BAR_MS = 300_000
DEFAULT_PRICE_PER_MTOK = 0.042
DEFAULT_EQUITY = 3044.35
IL = ZoneInfo("Asia/Jerusalem")
TESTNET_MISSING_DEFAULT = ("CLUSDT", "MUUSDT", "SNDKUSDT")


# --- request building (production code paths) ------------------------------------

def encode_body(payload: dict[str, Any]) -> str:
    """JSON body as typesafe_sdk.system_one sends it (state, model, questions)."""
    try:
        from typesafe_sdk._core.questions import normalize_questions
        from typesafe_sdk._core.transport import _encode_body

        body = {"state": payload["state"], "model": payload["model"], "questions": normalize_questions(payload["questions"])}
        return _encode_body(body).decode("utf-8")
    except Exception:  # noqa: BLE001 - SDK internals moved: plain JSON of the same objects
        qs = {k: (q.model_dump() if hasattr(q, "model_dump") else dict(q)) for k, q in payload["questions"].items()}
        return json.dumps({"state": payload["state"], "model": payload["model"], "questions": qs}, ensure_ascii=False,
                          separators=(",", ":"))


def build_request(symbol: str, candles: tuple[Candle, ...], equity: float) -> tuple[Any, dict[str, Any], str]:
    snap = snapshot_from_closed_bars(symbol, candles, tf="5m", book=None,
                                     position=Position(side="FLAT", size=0.0, cash_usdt=equity))
    compact = build_compact_state(snap, compute_features(snap))
    payload = build_system_one_payload(compact, allowed_actions=allowed_actions_for_position("FLAT"))
    return compact, payload, encode_body(payload)


# --- candles ------------------------------------------------------------------------

def load_candles(venue: str, symbol: str, start_ms: int, end_ms: int, cache: Path) -> list[Candle]:
    """Closed 5m klines with volume, public endpoint via curl (backtest_replay._get), cached."""
    import backtest_replay as b

    def fetch() -> list:
        out: list = []
        cur = start_ms
        while cur < end_ms:
            rows = b._get(f"{b.HOSTS[venue]}/fapi/v1/klines?symbol={symbol}&interval=5m&startTime={cur}"
                          f"&endTime={end_ms - 1}&limit=1500")
            if not rows:
                break
            out.extend(rows)
            nxt = int(rows[-1][0]) + BAR_MS
            if nxt <= cur:
                break
            cur = nxt
            time.sleep(0.1)
        return out

    try:
        rows = b._cached(cache, f"kv_{venue}_{symbol}_{start_ms}_{end_ms}.json", fetch)
    except Exception as exc:  # noqa: BLE001
        print(f"klines {venue} {symbol}: {exc}", file=sys.stderr)
        return []
    seen: dict[int, Candle] = {}
    for r in rows:
        t = int(r[0])
        if start_ms <= t < end_ms:
            seen[t] = Candle(ts=t, open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]), volume=float(r[5]))
    return [seen[t] for t in sorted(seen)]


# --- counting ---------------------------------------------------------------------------

def in_no_entry_window(bar_open_ms: int) -> bool:
    """Production guard 03:00-09:00 Asia/Jerusalem, checked at the call time (bar close)."""
    hour = datetime.fromtimestamp((bar_open_ms + BAR_MS) / 1000, tz=timezone.utc).astimezone(IL).hour
    return 3 <= hour < 9


def load_tokenizer(path: str | None) -> tuple[Any, str | None]:
    if path == "none":
        return None, None
    cands = [path] if path else sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/*/typed-decisions/tokenizer/tokenizer.json")))
    for p in cands:
        try:
            from tokenizers import Tokenizer

            return Tokenizer.from_file(p), p
        except Exception:  # noqa: BLE001
            continue
    return None, None


def count_symbol(symbol: str, candles: list[Candle], start_ms: int, end_ms: int, window: int, equity: float,
                 tokenizer: Any = None, min_bars: int = 50, samples: int = 0) -> dict[str, Any]:
    """Every closed bar in [start_ms, end_ms) with >= min_bars of history is one call."""
    calls = calls_outside_window = 0
    chars = chars_state = chars_outside = 0
    tokens = tokens_outside = 0
    per_call_chars: list[int] = []
    sample_bodies: list[dict[str, Any]] = []
    texts: list[str] = []
    flags: list[bool] = []
    for j, c in enumerate(candles):
        if not (start_ms <= c.ts < end_ms) or j + 1 < min_bars:
            continue
        win = tuple(candles[max(0, j + 1 - window): j + 1])
        compact, payload, body = build_request(symbol, win, equity)
        outside = not in_no_entry_window(c.ts)
        calls += 1
        chars += len(body)
        chars_state += len(json.dumps(payload["state"], ensure_ascii=False, separators=(",", ":")))
        per_call_chars.append(len(body))
        if outside:
            calls_outside_window += 1
            chars_outside += len(body)
        if tokenizer is not None:
            texts.append(body)
            flags.append(outside)
        if len(sample_bodies) < samples and j % 97 == 0:
            sample_bodies.append({"symbol": symbol, "bar_utc": iso(c.ts), "state_text": compact.as_text(), "body": body})
    if tokenizer is not None and texts:
        for k in range(0, len(texts), 2048):
            enc = tokenizer.encode_batch(texts[k:k + 2048])
            for e, fl in zip(enc, flags[k:k + 2048]):
                tokens += len(e.ids)
                if fl:
                    tokens_outside += len(e.ids)
    return {
        "symbol": symbol, "calls": calls, "calls_outside_window": calls_outside_window, "chars": chars,
        "chars_state_json": chars_state, "chars_outside_window": chars_outside,
        "tokens": tokens if tokenizer is not None else None,
        "tokens_outside_window": tokens_outside if tokenizer is not None else None,
        "chars_min": min(per_call_chars) if per_call_chars else None,
        "chars_max": max(per_call_chars) if per_call_chars else None,
        "samples": sample_bodies,
    }


def _worker(args: tuple) -> dict[str, Any]:
    venue, symbol, fetch_start, start_ms, end_ms, cache, window, equity, tok_path, samples = args
    candles = load_candles(venue, symbol, fetch_start, end_ms, Path(cache))
    tok, _ = load_tokenizer(tok_path)
    res = count_symbol(symbol, candles, start_ms, end_ms, window, equity, tok, samples=samples)
    res["bars_nominal"] = (end_ms - start_ms) // BAR_MS
    return res


def cost(tokens: float, price_per_mtok: float) -> float:
    return tokens / 1e6 * price_per_mtok


def summarize(rows: list[dict[str, Any]], symbols: Iterable[str], price: float, days: float, label: str) -> dict[str, Any]:
    sel = [r for r in rows if r["symbol"] in set(symbols)]
    calls = sum(r["calls"] for r in sel)
    nominal = sum(r["bars_nominal"] for r in sel)
    chars = sum(r["chars"] for r in sel)
    have_tok = all(r["tokens"] is not None for r in sel) and sel
    tok = sum(r["tokens"] for r in sel) if have_tok else None
    out: dict[str, Any] = {
        "label": label, "symbols": len(sel), "days": days, "calls": calls, "calls_nominal": nominal,
        "chars_total": chars, "chars_per_call": chars / calls if calls else None,
        "tokens_chars4": chars / 4, "tokens_chars3_5": chars / 3.5, "tokens_tokenizer": tok,
        "usd_chars4": cost(chars / 4, price), "usd_chars3_5": cost(chars / 3.5, price),
        "usd_tokenizer": cost(tok, price) if tok is not None else None,
        "calls_outside_window": sum(r["calls_outside_window"] for r in sel),
        "chars_outside_window": sum(r["chars_outside_window"] for r in sel),
        "tokens_tokenizer_outside_window": sum(r["tokens_outside_window"] for r in sel) if have_tok else None,
    }
    out["tokens_per_call_tokenizer"] = tok / calls if tok is not None and calls else None
    return out


def scaled(s: dict[str, Any], days_to: float) -> dict[str, Any]:
    k = days_to / s["days"]
    return {"days": days_to, "calls": s["calls"] * k,
            "tokens_tokenizer": None if s["tokens_tokenizer"] is None else s["tokens_tokenizer"] * k,
            "tokens_chars4": s["tokens_chars4"] * k, "tokens_chars3_5": s["tokens_chars3_5"] * k,
            "usd_tokenizer": None if s["usd_tokenizer"] is None else s["usd_tokenizer"] * k,
            "usd_chars4": s["usd_chars4"] * k, "usd_chars3_5": s["usd_chars3_5"] * k}


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def symbols_from_ledger(path: str) -> list[str]:
    import sqlite3

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = conn.execute("SELECT DISTINCT upper(symbol) FROM decisions WHERE judgment_json IS NOT NULL "
                        "AND json_extract(judgment_json, '$.model') NOT IN ('model_skipped')").fetchall()
    conn.close()
    return sorted(r[0] for r in rows)


# --- universe -------------------------------------------------------------------------------

def universe_from_ledger(path: str, day_from: str, day_to: str, size: int) -> tuple[list[str], list[dict[str, Any]]]:
    """Top `size` symbols by model answers (not model_skipped) on UTC days day_from..day_to (inclusive).

    Ties break alphabetically. Returns the symbols and the per-day answer counts of every symbol seen.
    """
    import sqlite3

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT upper(symbol), substr(ts, 1, 10), count(*) FROM decisions WHERE judgment_json IS NOT NULL "
        "AND json_extract(judgment_json, '$.model') NOT IN ('model_skipped') AND substr(ts, 1, 10) BETWEEN ? AND ? "
        "GROUP BY 1, 2", (day_from, day_to)).fetchall()
    conn.close()
    per: dict[str, dict[str, int]] = {}
    for sym, day, n in rows:
        per.setdefault(sym, {})[day] = n
    table = sorted(({"symbol": s, "total": sum(d.values()), "per_day": d} for s, d in per.items()),
                   key=lambda r: (-r["total"], r["symbol"]))
    return [r["symbol"] for r in table[:size]], table


# --- reconstruction check against logged states ------------------------------------------

FIELD_RE = re.compile(r"(\w+)=(\S+)")


def state_fields(text: str) -> dict[str, str]:
    """Named values of a state_text (ema_stack has no key= so it gets one)."""
    lines = text.split("\n")
    out = dict(FIELD_RE.findall(text))
    if len(lines) > 2 and lines[2].split():
        out["ema_stack"] = lines[2].split()[0]
    out["news"] = lines[-1] if lines and lines[-1].startswith("news:") else ""
    return out


def check_ledger(path: str, venue: str, cache: Path, n: int, seed: int = 2,
                 symbols: list[str] | None = None) -> dict[str, Any]:
    """Rebuild logged state_text from candles; exact-match rates overall and field by field.

    Compared with a fixed 300-bar window (what the month run uses) and with the best window in
    239..300 (the live buffer grows from 239 closed bars after a restart to 300). The flat-book
    fields (pos/size/cash_usdt) and the order book are expected to differ: the replay has no live
    position, wallet or depth.
    """
    import random
    import sqlite3

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = conn.execute("SELECT symbol, state_text FROM decisions WHERE judgment_json IS NOT NULL "
                        "AND json_extract(judgment_json, '$.model') NOT IN ('model_skipped') "
                        "AND state_text LIKE 'symbol=%'").fetchall()
    conn.close()
    want = {s.upper() for s in symbols} if symbols else None
    recs = []
    for sym, st in rows:
        mm = re.search(r"ts=(\S+)Z", st or "")
        if mm and (want is None or sym.upper() in want):
            t = int(datetime.strptime(mm.group(1), "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
            recs.append((sym.upper(), t, st))
    t0, t1 = min(r[1] for r in recs), max(r[1] for r in recs)
    sample = random.Random(seed).sample(recs, min(n, len(recs)))
    candles: dict[str, list[Candle]] = {}
    stats: dict[str, dict[str, list[int]]] = {"w300": {}, "best": {}}
    examples: dict[str, list[dict[str, str]]] = {}
    lines_w300 = lines_best = total = 0
    windows: dict[int, int] = {}
    for sym, t, st in sample:
        if sym not in candles:
            candles[sym] = load_candles(venue, sym, t0 - 320 * BAR_MS, t1 + BAR_MS, cache)
        cs = candles[sym]
        idx = {c.ts: i for i, c in enumerate(cs)}
        if t not in idx:
            continue
        j, logged = idx[t], st.split("\n")
        total += 1
        w300 = build_request(sym, tuple(cs[max(0, j - 299): j + 1]), DEFAULT_EQUITY)[0].as_text()
        best_text, best_w = w300, None
        for w in [300] + list(range(239, 300)):
            txt = w300 if w == 300 else build_request(sym, tuple(cs[max(0, j + 1 - w): j + 1]), DEFAULT_EQUITY)[0].as_text()
            if txt.split("\n")[1:4] == logged[1:4]:
                best_text, best_w = txt, w
                break
        lines_w300 += w300.split("\n")[1:4] == logged[1:4]
        lines_best += best_w is not None
        if best_w is not None:
            windows[best_w] = windows.get(best_w, 0) + 1
        lf = state_fields(st)
        for key, txt in (("w300", w300), ("best", best_text)):
            rf = state_fields(txt)
            for f, v in lf.items():
                cell = stats[key].setdefault(f, [0, 0])
                cell[0] += 1
                ok = rf.get(f) == v
                cell[1] += ok
                if key == "w300" and not ok and len(examples.setdefault(f, [])) < 3:
                    examples[f].append({"symbol": sym, "ts": iso(t), "logged": v, "rebuilt": str(rf.get(f))})
    rate = lambda d: {f: round(c[1] / c[0], 4) for f, c in sorted(d.items())}  # noqa: E731
    return {
        "venue": venue, "symbols": sorted(want) if want else "all", "checked": total,
        "market_lines_match_window_300": lines_w300 / total if total else None,
        "market_lines_match_some_window_239_300": lines_best / total if total else None,
        "field_match_window_300": rate(stats["w300"]), "field_match_best_window": rate(stats["best"]),
        "mismatch_examples_window_300": examples,
        "best_window_top": sorted(windows.items(), key=lambda kv: -kv[1])[:8],
    }


# --- plan, live client, budget guard, resume ----------------------------------------------

def build_plan(candles: dict[str, list[Candle]], start_ms: int, end_ms: int, min_bars: int = 50
               ) -> list[tuple[int, str, int]]:
    """(bar_ms, symbol, index) for every closed bar in range with >= min_bars history, bar-major order.

    Bar-major means an interrupted run still covers every symbol up to the same time.
    """
    plan = [(c.ts, sym, j) for sym, cs in candles.items() for j, c in enumerate(cs)
            if start_ms <= c.ts < end_ms and j + 1 >= min_bars]
    plan.sort()
    return plan


class BilledParseError(Exception):
    """The API answered (and billed) but the answer could not be parsed; never retried."""

    def __init__(self, usage: dict[str, Any], cause: Exception) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.usage = usage


def _usage_of(resp: Any) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    return {"input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None)}


class JevUsageClient:
    """Same request as jev_trader.jev.JevClient.judge, but also returns the SDK usage."""

    def __init__(self, api_key: str, client: Any = None) -> None:
        if client is None:
            from typesafe_sdk import TypeSafeClient

            from jev_trader import JEV_MODEL

            client = TypeSafeClient(api_key=api_key, model=JEV_MODEL)
        self._client = client

    def judge(self, compact: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
        from jev_trader.jev import judgment_from_response, position_side_from_compact

        allowed = allowed_actions_for_position(position_side_from_compact(compact))
        payload = build_system_one_payload(compact, allowed_actions=allowed)
        resp = self._client.system_one(state=payload["state"], questions=payload["questions"], model=payload["model"])
        try:
            judgment = judgment_from_response(resp)
        except Exception as exc:  # noqa: BLE001
            raise BilledParseError(_usage_of(resp), exc) from exc
        return judgment.raw, _usage_of(resp), judgment.model

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


@dataclass
class BudgetGuard:
    """Real-cost guard from SDK usage (input tokens x price; output tokens are recorded, not priced)."""

    budget_usd: float
    price_per_mtok: float
    planned_calls: int
    max_calls: int
    check_after: int = 2000
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_usage: int = 0

    def add(self, input_tokens: int | None, output_tokens: int | None, fallback_input: int) -> int:
        """Count one billed call; returns the input tokens charged to the budget (fallback if not reported)."""
        est = input_tokens is None
        charged = int(fallback_input if est else input_tokens)
        self.calls += 1
        self.input_tokens += charged
        self.output_tokens += int(output_tokens or 0)
        self.estimated_usage += est
        return charged

    def cost(self) -> float:
        return self.input_tokens / 1e6 * self.price_per_mtok

    def cost_per_call(self) -> float | None:
        return self.cost() / self.calls if self.calls else None

    def projection(self) -> float | None:
        cpc = self.cost_per_call()
        return None if cpc is None else cpc * self.planned_calls

    def stop_reason(self) -> str | None:
        """Checked before every call."""
        if self.calls >= self.max_calls:
            return f"max_calls reached ({self.calls} >= {self.max_calls})"
        if self.cost() >= self.budget_usd:
            return f"budget reached: real cost ${self.cost():.4f} >= ${self.budget_usd:.2f}"
        cpc = self.cost_per_call()
        if cpc is not None and self.cost() + cpc > self.budget_usd:
            return f"budget: the next call would pass ${self.budget_usd:.2f} (spent ${self.cost():.4f})"
        if self.calls >= self.check_after and (self.projection() or 0.0) > self.budget_usd:
            return (f"projection over budget: ${self.cost_per_call():.8f}/call x {self.planned_calls} planned calls = "
                    f"${self.projection():.2f} > ${self.budget_usd:.2f} (after {self.calls} calls, spent ${self.cost():.4f})")
        if self.estimated_usage > 20:
            return f"usage missing in {self.estimated_usage} responses; the budget cannot be enforced"
        return None


def load_billed_errors(path: Path) -> list[dict[str, Any]]:
    """Error lines that were billed (the API answered); they count toward the budget on resume."""
    out: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("billed"):
                    out.append(rec)
    return out


def seed_guard(guard: "BudgetGuard", answers: Path) -> None:
    """Resume: charge every earlier answer and billed error to the guard."""
    _, prior = load_done(answers)
    for rec in prior + load_billed_errors(answers.with_suffix(".errors.jsonl")):
        u = rec.get("usage") or {}
        guard.add(rec.get("budget_input_tokens", u.get("input_tokens")), u.get("output_tokens"), fallback_input=0)


def load_done(path: Path) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    """Answered (symbol, bar_ms) keys from an append-only JSONL; a torn last line is ignored."""
    done: set[tuple[str, int]] = set()
    recs: list[dict[str, Any]] = []
    if not path.is_file():
        return done, recs
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("judgment") and "symbol" in rec and "bar_ms" in rec:
                key = (str(rec["symbol"]), int(rec["bar_ms"]))
                if key not in done:
                    done.add(key)
                    recs.append(rec)
    return done, recs


NON_RETRYABLE = ("TypeSafeAuthenticationError", "TypeSafePermissionDeniedError", "TypeSafeBadRequestError",
                 "TypeSafeNotFoundError", "TypeSafeUnprocessableEntityError", "TypeSafeError")


def run_live(plan: list[tuple[int, str, int]], candles: dict[str, list[Candle]], client: Any, answers: Path,
             guard: BudgetGuard, *, window: int, equity: float, min_interval: float = 0.2, attempts: int = 3,
             max_consecutive_errors: int = 5, sleep=time.sleep, log=print) -> dict[str, Any]:
    """Ask the model for every planned bar not yet in `answers`; append one JSON line per answer.

    The guard must already hold the usage of earlier answers (resume). Stops on the guard, on a
    non-retryable API error, or after max_consecutive_errors bars that failed every attempt.
    """
    done, _ = load_done(answers)
    errors_path = answers.with_suffix(".errors.jsonl")
    todo = [p for p in plan if (p[1], p[0]) not in done]
    log(f"plan {len(plan)} calls, already answered {len(plan) - len(todo)}, to do {len(todo)}; "
        f"spent so far ${guard.cost():.4f} on {guard.calls} calls")
    stop, consecutive, new_calls, last = None, 0, 0, 0.0
    for path in (answers, errors_path):  # a crash can leave a torn last line: start the next record on a new line
        if path.is_file() and path.stat().st_size:
            with path.open("rb") as fb:
                fb.seek(-1, os.SEEK_END)
                torn = fb.read(1) != b"\n"
            if torn:
                with path.open("a", encoding="utf-8") as fa:
                    fa.write("\n")
    with answers.open("a", encoding="utf-8") as fh, errors_path.open("a", encoding="utf-8") as eh:
        for bar_ms, sym, j in todo:
            stop = guard.stop_reason()
            if stop:
                break
            cs = candles[sym]
            win = tuple(cs[max(0, j + 1 - window): j + 1])
            compact, _payload, body = build_request(sym, win, equity)
            result, err = None, None
            fallback = math.ceil(len(body) / 2)  # conservative: the proxy tokenizer gives ~2.2 chars/token
            for attempt in range(attempts):
                wait = min_interval - (time.monotonic() - last)
                if wait > 0:
                    sleep(wait)
                last = time.monotonic()
                t_call = time.perf_counter()
                try:
                    result = client.judge(compact)
                    break
                except Exception as exc:  # noqa: BLE001
                    err = exc
                    name = type(exc).__name__
                    billed = isinstance(exc, BilledParseError) or name == "TypeSafeAPIResponseValidationError"
                    rec = {"symbol": sym, "bar_ms": bar_ms, "attempt": attempt + 1, "error": name,
                           "status": getattr(exc, "status", None), "billed": billed, "at": iso(int(time.time() * 1000))}
                    if billed:  # the server answered: charge it, never pay twice for the bar
                        usage = getattr(exc, "usage", None) or {}
                        rec["usage"] = usage
                        rec["budget_input_tokens"] = guard.add(usage.get("input_tokens"), usage.get("output_tokens"),
                                                               fallback_input=fallback)
                    eh.write(json.dumps(rec) + "\n")
                    eh.flush()
                    if billed:
                        break
                    if name in NON_RETRYABLE:
                        stop = f"non-retryable API error {name} (status {getattr(exc, 'status', None)})"
                        break
                    sleep(2.0 * (2 ** attempt))
            if stop:
                break
            if result is None:
                consecutive += 1
                if consecutive >= max_consecutive_errors:
                    stop = f"{consecutive} bars in a row failed (last error {type(err).__name__})"
                    break
                continue
            consecutive = 0
            raw, usage, model = result
            charged = guard.add(usage.get("input_tokens"), usage.get("output_tokens"), fallback_input=fallback)
            fh.write(json.dumps({"symbol": sym, "bar_ms": bar_ms, "bar_utc": iso(bar_ms), "model": model,
                                 "judgment": raw, "usage": usage,
                                 "usage_estimated": usage.get("input_tokens") is None, "budget_input_tokens": charged,
                                 "latency_ms": round((time.perf_counter() - t_call) * 1000, 1),
                                 "answered_at": iso(int(time.time() * 1000))}, ensure_ascii=False) + "\n")
            fh.flush()
            new_calls += 1
            if new_calls % 100 == 0:
                os.fsync(fh.fileno())
            if new_calls % 1000 == 0:
                log(f"{new_calls} new calls, total {guard.calls}, spent ${guard.cost():.4f}, "
                    f"projection ${guard.projection() or 0:.2f}, at {iso(bar_ms)}")
    summary = {"stop_reason": stop or "plan complete", "new_calls": new_calls, "total_calls": guard.calls,
               "input_tokens": guard.input_tokens, "output_tokens": guard.output_tokens,
               "cost_usd": round(guard.cost(), 6), "projection_usd": guard.projection(),
               "estimated_usage_responses": guard.estimated_usage}
    log(("STOPPED: " if stop else "DONE: ") + json.dumps(summary))
    return summary


# --- main ---------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", default=None, help="UTC date (exclusive), default: today 00:00 UTC")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--symbols", default="", help="comma list")
    ap.add_argument("--symbols-from-ledger", default=None, help="read-only ledger copy; its answered symbols")
    ap.add_argument("--universe-ledger", default=None, metavar="LEDGER",
                    help="read-only ledger copy: take the top --universe-size symbols by model answers on "
                         "--universe-days")
    ap.add_argument("--universe-days", default="2026-09-21,2026-09-23", help="first,last UTC day (inclusive)")
    ap.add_argument("--universe-size", type=int, default=15)
    ap.add_argument("--testnet-missing", default=",".join(TESTNET_MISSING_DEFAULT),
                    help="symbols without testnet markets (reported as a second subset)")
    ap.add_argument("--candle-venue", default="mainnet", choices=("mainnet", "testnet"))
    ap.add_argument("--cache", default="/tmp/jevbt/cache")
    ap.add_argument("--window", type=int, default=300, help="closed bars per state (live buffer max 300)")
    ap.add_argument("--equity", type=float, default=DEFAULT_EQUITY, help="cash_usdt shown in the flat state")
    ap.add_argument("--price-per-mtok", type=float, default=DEFAULT_PRICE_PER_MTOK, help="USD per 1M input tokens")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json path, or 'none'")
    ap.add_argument("--samples", type=int, default=2, help="sample requests per symbol written to the out dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="/tmp/jevbt/out/month")
    ap.add_argument("--dry-run", action="store_true", default=True, help="default; never calls the model")
    ap.add_argument("--live", action="store_true", help="paid Jev calls (also needs --i-have-approval)")
    ap.add_argument("--i-have-approval", action="store_true")
    ap.add_argument("--max-calls", type=int, default=0, help="hard cap for --live (all answers in the file count)")
    ap.add_argument("--budget-usd", type=float, default=4.0, help="hard cap on real cost from SDK usage")
    ap.add_argument("--check-after", type=int, default=2000, help="calls before the projection check starts")
    ap.add_argument("--answers", default=None, help="append-only answers JSONL (default <out>/live_answers.jsonl)")
    ap.add_argument("--min-interval", type=float, default=0.2, help="seconds between call starts")
    ap.add_argument("--check-ledger", default=None, metavar="LEDGER",
                    help="only compare rebuilt states with logged state_text (read-only ledger copy)")
    ap.add_argument("--check-n", type=int, default=150)
    ap.add_argument("--tag", default="", help="suffix for the dry-run output file names")
    return ap.parse_args(argv)


def resolve_symbols(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]] | None]:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    table = None
    if args.symbols_from_ledger:
        symbols = sorted(set(symbols) | set(symbols_from_ledger(args.symbols_from_ledger)))
    if args.universe_ledger:
        d0, d1 = (x.strip() for x in args.universe_days.split(","))
        uni, table = universe_from_ledger(args.universe_ledger, d0, d1, args.universe_size)
        symbols = sorted(set(symbols) | set(uni))
    return symbols, table


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = parse_args(argv)
    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end
           else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0))
    end_ms = int(end.timestamp() * 1000)
    start_ms = int((end - timedelta(days=args.days)).timestamp() * 1000)
    fetch_start = start_ms - args.window * BAR_MS
    symbols, universe_table = resolve_symbols(args)
    if not symbols:
        raise SystemExit("no symbols (use --symbols, --symbols-from-ledger or --universe-ledger)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.cache).mkdir(parents=True, exist_ok=True)

    if args.check_ledger:
        res = check_ledger(args.check_ledger, args.candle_venue, Path(args.cache), args.check_n, symbols=symbols)
        (out / f"state_check_{args.candle_venue}_{len(symbols)}sym.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return res
    if args.live:
        if not args.i_have_approval or args.max_calls <= 0 or args.budget_usd <= 0:
            raise SystemExit("--live needs --i-have-approval, --max-calls > 0 and --budget-usd > 0; nothing was called")
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            raise SystemExit("TYPESAFE_API_KEY is not set in this process environment; nothing was called")
        candles = {s: load_candles(args.candle_venue, s, fetch_start, end_ms, Path(args.cache)) for s in symbols}
        plan = build_plan(candles, start_ms, end_ms)
        answers = Path(args.answers) if args.answers else out / "live_answers.jsonl"
        guard = BudgetGuard(args.budget_usd, args.price_per_mtok, planned_calls=len(plan), max_calls=args.max_calls,
                            check_after=args.check_after)
        seed_guard(guard, answers)
        client = JevUsageClient(api_key=key)
        del key
        try:
            summary = run_live(plan, candles, client, answers, guard, window=args.window, equity=args.equity,
                               min_interval=args.min_interval)
        finally:
            client.close()
        (out / "live_status.json").write_text(json.dumps(summary, indent=1))
        return summary

    _, tok_path = load_tokenizer(args.tokenizer)
    tasks = [(args.candle_venue, s, fetch_start, start_ms, end_ms, args.cache, args.window, args.equity,
              tok_path or "none", args.samples) for s in symbols]
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_worker, tasks))
    else:
        rows = [_worker(t) for t in tasks]
    missing = {s.strip().upper() for s in args.testnet_missing.split(",") if s.strip()}
    full = summarize(rows, symbols, args.price_per_mtok, args.days, f"all {len(symbols)} symbols")
    tn = summarize(rows, [s for s in symbols if s not in missing], args.price_per_mtok, args.days,
                   f"testnet {len([s for s in symbols if s not in missing])} symbols")
    result = {
        "mode": "dry-run (no model calls)",
        "range_utc": [iso(start_ms), iso(end_ms)], "candle_venue": args.candle_venue, "window": args.window,
        "equity": args.equity, "price_per_mtok": args.price_per_mtok, "tokenizer": tok_path,
        "tokenizer_note": "local Laya tokenizer used as a proxy; Jev's own tokenizer and server prompt are unknown",
        "elapsed_sec": round(time.time() - t0, 1),
        "symbols": symbols, "universe_rule": None if universe_table is None else
        f"top {args.universe_size} by model answers on {args.universe_days} (UTC, inclusive)",
        "universe_counts": universe_table,
        "planned_calls": full["calls"], "budget_usd": args.budget_usd,
        "summary": [full, tn],
        "scaled": {s["label"]: [scaled(s, 31), scaled(s, 60)] for s in (full, tn)},
        "per_symbol": [{k: v for k, v in r.items() if k != "samples"} for r in rows],
    }
    samples = [smp for r in rows for smp in r["samples"]]
    (out / f"sample_requests{args.tag}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in samples) + "\n")
    (out / f"month_dry_run{args.tag}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (out / f"month_dry_run{args.tag}.md").write_text(render(result))
    print(render(result))
    return result


def render(r: dict[str, Any]) -> str:
    L = [f"# Jev month replay, DRY RUN: {r['range_utc'][0]} .. {r['range_utc'][1]} ({r['candle_venue']} candles, "
         f"window {r['window']}), ${r['price_per_mtok']}/1M input tokens", "",
         f"tokenizer: {r['tokenizer']} ({r['tokenizer_note']})", "",
         "| set | days | calls (nominal) | chars/call | tok/call (tokenizer) | tokens tokenizer | $ tokenizer | "
         "tokens chars/4 | $ chars/4 | tokens chars/3.5 | $ chars/3.5 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]

    def fm(x: Any, d: int = 0) -> str:
        return "n/a" if x is None else f"{x:,.{d}f}"

    for s in r["summary"]:
        L.append(f"| {s['label']} | {s['days']:g} | {s['calls']:,} ({s['calls_nominal']:,}) | {fm(s['chars_per_call'], 1)} | "
                 f"{fm(s['tokens_per_call_tokenizer'], 1)} | {fm(s['tokens_tokenizer'])} | {fm(s['usd_tokenizer'], 2)} | "
                 f"{fm(s['tokens_chars4'])} | {fm(s['usd_chars4'], 2)} | {fm(s['tokens_chars3_5'])} | {fm(s['usd_chars3_5'], 2)} |")
        for sc in r["scaled"][s["label"]]:
            L.append(f"| {s['label']} (scaled) | {sc['days']:g} | {sc['calls']:,.0f} | | | {fm(sc['tokens_tokenizer'])} | "
                     f"{fm(sc['usd_tokenizer'], 2)} | {fm(sc['tokens_chars4'])} | {fm(sc['usd_chars4'], 2)} | "
                     f"{fm(sc['tokens_chars3_5'])} | {fm(sc['usd_chars3_5'], 2)} |")
    L.append("")
    for s in r["summary"]:
        tok_o = s["tokens_tokenizer_outside_window"]
        L.append(f"- {s['label']}: outside the 03-09 Asia/Jerusalem no-entry window {s['calls_outside_window']:,} calls, "
                 f"tokens tokenizer {fm(tok_o)} (${fm(None if tok_o is None else cost(tok_o, r['price_per_mtok']), 2)}), "
                 f"chars/4 {s['chars_outside_window'] / 4:,.0f} (${cost(s['chars_outside_window'] / 4, r['price_per_mtok']):.2f})")
    full = r["summary"][0]
    L.append(f"\nsymbols ({len(r['symbols'])}): {', '.join(r['symbols'])}" +
             (f"; rule: {r['universe_rule']}" if r.get("universe_rule") else ""))
    L.append(f"planned calls {r['planned_calls']:,}; budget ${r['budget_usd']:.2f}: tokenizer estimate "
             f"${fm(full['usd_tokenizer'], 2)}, chars/4 ${full['usd_chars4']:.2f}, chars/3.5 ${full['usd_chars3_5']:.2f}; "
             f"break-even input tokens per call for the budget: "
             f"{r['budget_usd'] / r['price_per_mtok'] * 1e6 / max(1, r['planned_calls']):,.1f}")
    L.append(f"\nelapsed {r['elapsed_sec']} s; no model calls were made")
    return "\n".join(L)


if __name__ == "__main__":
    main()
