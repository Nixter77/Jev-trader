#!/usr/bin/env python3
"""Rebuild the bot's Jev requests for a historical range and estimate calls, tokens and cost.

DRY RUN IS THE DEFAULT: no model call is made, no API key is read. The live path
exists for a later, approved run only; it needs BOTH `--live` and
`--i-have-approval` plus --max-calls. The key (never printed or logged) comes
from the process environment (TYPESAFE_API_KEY, which typesafe_sdk reads), or
with --env-file only that key is read from the file with the bot's own loader
(python-dotenv + map_typesafe_api_key); nothing else from the file is used. It:
  - asks one FLAT buy question per planned bar, bar-major (all symbols advance
    together), with --workers parallel threads (default 4) and --min-interval
    seconds between any two call starts;
  - appends every answer (raw judgment + SDK usage input/output tokens) to an
    append-only JSONL (single writer under a lock); a rerun skips bars already
    answered or billed-but-unparsable (resume, no double pay); the plan
    (symbols, range, window) is pinned in <answers>.meta.json and a resume
    with a different plan is refused; a log goes to <answers>.log;
  - budget guard from real usage at --price-per-mtok: a call slot is reserved
    (worst-case tokens) before sending and settled from usage afterwards, so
    --max-calls and --budget-usd hold across workers; after --check-after
    completed calls (2000) it projects cost/call x planned calls and stops if
    that exceeds --budget-usd; it also stops if usage is missing;
  - 429/418 rate limits pause ALL workers (Retry-After respected, else
    exponential backoff); transport errors: 3 attempts with backoff; a billed
    but unparsable answer is charged and never retried; auth/permission/bad-
    request errors stop everything, so do 5 failed bars in a row; Ctrl-C stops
    issuing calls and saves the in-flight answers;
  - --import-from OLD: a NEW answers file starts with the in-plan answers (and
    billed-but-unparsable records) of an earlier run with the same range /
    venue / window / equity (old files only read, sha256 recorded); every
    other call the old run paid for is `prior_spend_usd` in the new meta and
    counts against the TOTAL cap; --import-only does just this (no key);
  - the projection (prior + spent + remaining x max(average so far, average of
    the last 2000 calls)) is re-checked before every call after --check-after,
    so a drift in tokens/call stops the run early and cheaply.

Pre-run decisions (2026-09-28, see PRE_RUN_DECISIONS): first 14 symbols (top-15
by model answers 21-23.09 UTC minus SKHYNIXUSDT); after the guard stopped that
run at 2001 calls, the first 9 of the same ranking, $4 total cap incl. the
$0.1007 already spent.

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
import hashlib
import json
import math
import os
import re
import random
import statistics
import sys
import threading
import time
from collections import deque
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
# Decisions fixed BEFORE the paid run (also printed in the month_eval report header).
PRE_RUN_DECISIONS = (
    "Universe: top-15 by model answers 21-23.09 UTC minus SKHYNIXUSDT (fewest answers), decided 2026-09-28 before "
    "the paid run to fit the $4 cap (14 symbols, --exclude SKHYNIXUSDT); evaluation rule unchanged.",
    "After the guard stopped the 14-symbol run at 2001 calls (real ~1198 input tokens/call, projection $6.08), the "
    "universe was cut on 2026-09-28 to the first 9 of the same pre-fixed ranking (top by model answers 21-23.09 UTC): "
    "SOLUSDT, BTCUSDT, ETHUSDT, ZECUSDT, XRPUSDT, XAUUSDT, DOGEUSDT, CLUSDT, NEARUSDT. The choice does not depend on "
    "the answers already received (they were not inspected); the evaluation rule is unchanged; the cap is $4 TOTAL, "
    "including the $0.1007 already spent by the stopped run (answers of the 9 symbols are imported, not re-paid).",
)


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

            from typesafe_sdk import RetryPolicy

            # no SDK-internal retries: run_live owns retries and the rate-limit pause shared by all workers
            client = TypeSafeClient(api_key=api_key, model=JEV_MODEL, retry=RetryPolicy(max_retries=0))
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
    inflight: int = 0
    inflight_tokens: int = 0
    max_input_seen: int = 0
    prior_usd: float = 0.0  # spent earlier on calls that are NOT in this plan (counts against the total cap)
    recent_window: int = 2000
    recent: Any = None  # deque of the last recent_window charged input-token counts

    def __post_init__(self) -> None:
        if self.recent is None:
            self.recent = deque(maxlen=max(1, self.recent_window))

    def add(self, input_tokens: int | None, output_tokens: int | None, fallback_input: int) -> int:
        """Count one billed call; returns the input tokens charged to the budget (fallback if not reported)."""
        est = input_tokens is None
        charged = int(fallback_input if est else input_tokens)
        self.calls += 1
        self.input_tokens += charged
        self.output_tokens += int(output_tokens or 0)
        self.estimated_usage += est
        self.max_input_seen = max(self.max_input_seen, charged)
        self.recent.append(charged)
        return charged

    def worst_case_tokens(self, fallback_input: int) -> int:
        """Tokens reserved for a call before it is sent: max(chars/2 of its body, 1.5 x largest input seen)."""
        return max(int(fallback_input), math.ceil(1.5 * self.max_input_seen))

    def reserve(self, worst_tokens: int) -> str | None:
        """Reserve one call slot (caller holds the run lock); returns a stop reason instead if it may not be sent.

        Completed calls drive stop_reason (incl. the projection); in-flight calls count at their worst case,
        so concurrent workers can never together pass --max-calls or the budget.
        """
        reason = self.stop_reason()
        if reason:
            return reason
        if self.calls + self.inflight >= self.max_calls:
            return f"max_calls reached ({self.calls} done + {self.inflight} in flight >= {self.max_calls})"
        worst = self.prior_usd + (self.input_tokens + self.inflight_tokens + worst_tokens) / 1e6 * self.price_per_mtok
        if worst > self.budget_usd:
            return (f"budget: spent ${self.total_cost():.4f} + {self.inflight + 1} reserved call(s) at worst case "
                    f"${worst:.4f} would pass ${self.budget_usd:.2f}")
        self.inflight += 1
        self.inflight_tokens += worst_tokens
        return None

    def release(self, worst_tokens: int) -> None:
        """Give a reserved slot back (the call was not billed)."""
        self.inflight -= 1
        self.inflight_tokens -= worst_tokens

    def settle(self, worst_tokens: int, input_tokens: int | None, output_tokens: int | None, fallback_input: int) -> int:
        """Turn a reservation into a billed call with the real usage."""
        self.release(worst_tokens)
        return self.add(input_tokens, output_tokens, fallback_input)

    def cost(self) -> float:
        """Real cost of the calls of this plan (answers + billed errors in the answers files)."""
        return self.input_tokens / 1e6 * self.price_per_mtok

    def total_cost(self) -> float:
        """Everything spent against the cap: prior out-of-plan spend + this plan."""
        return self.prior_usd + self.cost()

    def cost_per_call(self) -> float | None:
        return self.cost() / self.calls if self.calls else None

    def recent_cost_per_call(self) -> float | None:
        return sum(self.recent) / len(self.recent) / 1e6 * self.price_per_mtok if self.recent else None

    def projection(self) -> float | None:
        """Total at the end of the plan: prior + spent + remaining calls x max(average so far, recent average).

        The recent window (last recent_window calls) makes a drift in tokens/call visible early.
        """
        cpc = self.cost_per_call()
        if cpc is None:
            return None
        per = max(cpc, self.recent_cost_per_call() or 0.0)
        return self.total_cost() + per * max(0, self.planned_calls - self.calls)

    def stop_reason(self) -> str | None:
        """Checked before every call (so the projection is re-checked continuously after check_after calls)."""
        if self.calls >= self.max_calls:
            return f"max_calls reached ({self.calls} >= {self.max_calls})"
        if self.total_cost() >= self.budget_usd:
            return f"budget reached: real cost ${self.total_cost():.4f} >= ${self.budget_usd:.2f}"
        cpc = self.cost_per_call()
        if cpc is not None and self.total_cost() + cpc > self.budget_usd:
            return f"budget: the next call would pass ${self.budget_usd:.2f} (spent ${self.total_cost():.4f})"
        if self.calls >= self.check_after and (self.projection() or 0.0) > self.budget_usd:
            per = max(cpc or 0.0, self.recent_cost_per_call() or 0.0)
            return (f"projection over budget: prior ${self.prior_usd:.4f} + spent ${self.cost():.4f} + "
                    f"{max(0, self.planned_calls - self.calls)} remaining x ${per:.8f}/call = ${self.projection():.2f} > "
                    f"${self.budget_usd:.2f} (after {self.calls} of {self.planned_calls} planned calls)")
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


PLAN_FIXED_KEYS = ("range_utc", "candle_venue", "window", "equity")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def charged_tokens(rec: dict[str, Any]) -> int:
    u = rec.get("usage") or {}
    return int(rec.get("budget_input_tokens") or u.get("input_tokens") or 0)


def import_prior(old_answers: Path, new_answers: Path, plan_keys: set[tuple[str, int]], new_meta: dict[str, Any],
                 price_per_mtok: float) -> dict[str, Any]:
    """Copy the answers (and billed-but-unparsable records) of an earlier run that fall inside the new plan.

    The old files are only read. The old meta must match the new plan (range, candle venue, window, equity).
    Everything the old run paid for but that is NOT imported becomes `prior_spend_usd` (counts against the cap);
    the imported records keep their usage and are charged when the guard is seeded from the new file.
    """
    old_meta_path = old_answers.with_suffix(".meta.json")
    if not old_answers.is_file() or not old_meta_path.is_file():
        raise SystemExit(f"--import-from {old_answers}: answers or meta file missing; nothing was imported")
    old_meta = json.loads(old_meta_path.read_text())
    diff = [k for k in PLAN_FIXED_KEYS if old_meta.get(k) != new_meta.get(k)]
    if diff:
        raise SystemExit(f"--import-from refused: {', '.join(diff)} differ between the old and the new plan")
    if new_answers.exists() and new_answers.stat().st_size:
        raise SystemExit(f"--import-from refused: {new_answers} already has records (import only into a new file)")
    sha_before = sha256_of(old_answers)
    _, old_recs = load_done(old_answers)
    old_billed = load_billed_errors(old_answers.with_suffix(".errors.jsonl"))
    keep = [r for r in old_recs if (str(r["symbol"]), int(r["bar_ms"])) in plan_keys]
    kept_keys = {(str(r["symbol"]), int(r["bar_ms"])) for r in keep}
    keep_billed, seen = [], set()
    for r in old_billed:
        k = (str(r.get("symbol")), int(r.get("bar_ms") or 0))
        if k in plan_keys and k not in kept_keys and k not in seen:
            seen.add(k)
            keep_billed.append(r)
    all_tok = sum(charged_tokens(r) for r in old_recs + old_billed)
    imp_tok = sum(charged_tokens(r) for r in keep + keep_billed)
    new_answers.parent.mkdir(parents=True, exist_ok=True)
    for path, recs in ((new_answers, keep), (new_answers.with_suffix(".errors.jsonl"), keep_billed)):
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps({**r, "imported_from": str(old_answers)}, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    if sha256_of(old_answers) != sha_before:
        raise SystemExit("the old answers file changed during the import; aborting")
    return {"source": str(old_answers), "source_sha256": sha_before, "source_symbols": old_meta.get("symbols"),
            "source_calls": len(old_recs) + len(old_billed), "source_input_tokens": all_tok,
            "prior_total_spend_usd": round(all_tok / 1e6 * price_per_mtok, 6),
            "imported_answers": len(keep), "imported_billed_unparsable": len(keep_billed),
            "imported_input_tokens": imp_tok, "imported_spend_usd": round(imp_tok / 1e6 * price_per_mtok, 6),
            "prior_spend_usd": round((all_tok - imp_tok) / 1e6 * price_per_mtok, 9),
            "imported_at": iso(int(time.time() * 1000))}


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
RATE_LIMIT_STATUSES = (429, 418)


def is_rate_limited(exc: BaseException) -> bool:
    return type(exc).__name__ == "TypeSafeRateLimitError" or getattr(exc, "status", None) in RATE_LIMIT_STATUSES


def retry_after_seconds(exc: BaseException) -> float | None:
    """Retry-After of a rate-limit error in seconds (SDK retry_after_ms, retry-after-ms or retry-after headers)."""
    ms_ = getattr(exc, "retry_after_ms", None)
    if isinstance(ms_, (int, float)) and ms_ >= 0:
        return float(ms_) / 1000
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None

    def get(name: str) -> str | None:
        try:
            v = headers.get(name)
            return v if v is not None else headers.get(name.title())
        except Exception:  # noqa: BLE001
            return None

    raw_ms, raw_s = get("retry-after-ms"), get("retry-after")
    try:
        if raw_ms is not None:
            return max(0.0, float(raw_ms) / 1000)
        if raw_s is not None:
            return max(0.0, float(raw_s))
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime

            return max(0.0, parsedate_to_datetime(raw_s).timestamp() - time.time())
        except Exception:  # noqa: BLE001
            return None
    return None


class SharedPacer:
    """Pacing shared by all workers: --min-interval between any two call starts, and one common pause after a
    rate-limit answer (Retry-After if given, else exponential backoff with jitter), so every worker slows down."""

    def __init__(self, min_interval: float, base: float = 2.0, cap: float = 300.0, clock=time.monotonic) -> None:
        self.lock = threading.Lock()
        self.min_interval, self.base, self.cap, self.clock = min_interval, base, cap, clock
        self.next_start = 0.0
        self.pause_until = 0.0
        self.strikes = 0

    def wait_turn(self, stop: threading.Event) -> bool:
        """Block until this worker may start a call; False if the run is stopping."""
        while not stop.is_set():
            with self.lock:
                now = self.clock()
                t = max(now, self.next_start, self.pause_until)
                if t <= now:
                    self.next_start = now + self.min_interval
                    return True
            stop.wait(min(t - now, 0.5))
        return False

    def rate_limited(self, retry_after: float | None) -> float:
        with self.lock:
            self.strikes += 1
            if retry_after is not None:
                delay = retry_after
            else:
                delay = self.base * 2 ** min(self.strikes - 1, 16) * (1 + 0.25 * random.random())
            delay = min(delay, self.cap)
            self.pause_until = max(self.pause_until, self.clock() + delay)
            return delay

    def ok(self) -> None:
        with self.lock:
            self.strikes = 0


def _repair_torn(*paths: Path) -> None:
    """A crash can leave a torn last line: start the next record on a new line."""
    for path in paths:
        if path.is_file() and path.stat().st_size:
            with path.open("rb") as fb:
                fb.seek(-1, os.SEEK_END)
                torn = fb.read(1) != b"\n"
            if torn:
                with path.open("a", encoding="utf-8") as fa:
                    fa.write("\n")


def run_live(plan: list[tuple[int, str, int]], candles: dict[str, list[Candle]], client: Any, answers: Path,
             guard: BudgetGuard, *, window: int, equity: float, workers: int = 1, min_interval: float = 0.2,
             attempts: int = 3, rate_limit_attempts: int = 8, max_consecutive_errors: int = 5,
             backoff_base: float = 2.0, rate_limit_base: float = 2.0, retry_after_cap: float = 300.0,
             sleep=None, log=print) -> dict[str, Any]:
    """Ask the model for every planned bar not yet answered; append one JSON line per answer.

    `client` is either one client with .judge() shared by all workers, or a zero-argument factory called once
    per worker (each worker then closes its own client). The guard must already hold the usage of earlier
    answers (resume). Concurrency rules:
      - every bar is handed to exactly one worker (bars already answered, or billed-but-unparsable, are skipped);
      - a call slot is reserved in the guard under the run lock before sending (worst-case tokens), and settled
        with the real usage afterwards, so --max-calls and the budget hold across workers; the projection check
        uses completed calls;
      - one lock serializes guard, counters and both JSONL writers, so lines never interleave;
      - rate limits (429/418) pause ALL workers (SharedPacer, Retry-After respected) and do not count as
        attempts, up to rate_limit_attempts per bar; transport errors get `attempts` tries with backoff;
      - a billed but unparsable answer is charged and never retried; auth/permission/bad-request errors stop
        everything; max_consecutive_errors failed bars in a row (completion order) stop everything;
      - Ctrl-C stops issuing calls, waits for in-flight calls (already paid) and saves their answers.
    """
    done, _ = load_done(answers)
    errors_path = answers.with_suffix(".errors.jsonl")
    billed_keys = {(str(r.get("symbol")), int(r.get("bar_ms"))) for r in load_billed_errors(errors_path)
                   if r.get("symbol") is not None and r.get("bar_ms") is not None}
    todo = [p for p in plan if (p[1], p[0]) not in done and (p[1], p[0]) not in billed_keys]
    log(f"plan {len(plan)} calls, already answered {len(done)}, billed-unparsable {len(billed_keys)}, "
        f"to do {len(todo)}; spent so far ${guard.total_cost():.4f} (prior ${guard.prior_usd:.4f}) on {guard.calls} "
        f"in-plan calls; workers {workers}")
    _repair_torn(answers, errors_path)
    stop_evt = threading.Event()
    lock = threading.Lock()
    slots = threading.Condition(lock)
    st: dict[str, Any] = {"stop": None, "consecutive": 0, "new_calls": 0, "next": 0, "requests": 0,
                          "rate_limited": 0, "t0": time.monotonic()}
    pacer = SharedPacer(min_interval, rate_limit_base, retry_after_cap)
    nap = sleep or (lambda sec: stop_evt.wait(sec))
    shared = hasattr(client, "judge")
    fh = answers.open("a", encoding="utf-8")
    eh = errors_path.open("a", encoding="utf-8")

    def halt(reason: str) -> None:  # caller holds lock
        if st["stop"] is None:
            st["stop"] = reason
        stop_evt.set()

    def failed_bar(name: str) -> None:  # caller holds lock
        st["consecutive"] += 1
        if st["consecutive"] >= max_consecutive_errors:
            halt(f"{st['consecutive']} bars in a row failed (last error {name})")

    def write_error(rec: dict[str, Any]) -> None:  # caller holds lock
        eh.write(json.dumps(rec) + "\n")
        eh.flush()

    def ask(cl: Any, compact: Any, bar_ms: int, sym: str, fallback: int, worst: int) -> None:
        tries = rl = 0
        while True:
            if not pacer.wait_turn(stop_evt):
                with lock:
                    guard.release(worst)
                return
            t_call = time.perf_counter()
            with lock:
                st["requests"] += 1
            try:
                raw, usage, model = cl.judge(compact)
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                billed = isinstance(exc, BilledParseError) or name == "TypeSafeAPIResponseValidationError"
                rate = not billed and is_rate_limited(exc)
                rec = {"symbol": sym, "bar_ms": bar_ms, "attempt": tries + rl + 1, "error": name,
                       "status": getattr(exc, "status", None), "billed": billed, "rate_limited": rate,
                       "at": iso(int(time.time() * 1000))}
                if rate:
                    rec["pause_s"] = round(pacer.rate_limited(retry_after_seconds(exc)), 3)
                with lock:
                    if billed:  # the server answered: charge it, never pay twice for the bar
                        u = getattr(exc, "usage", None) or {}
                        rec["usage"] = u
                        rec["budget_input_tokens"] = guard.settle(worst, u.get("input_tokens"), u.get("output_tokens"),
                                                                  fallback)
                    write_error(rec)
                    if billed:
                        failed_bar(name)
                        return
                    if name in NON_RETRYABLE:
                        guard.release(worst)
                        halt(f"non-retryable API error {name} (status {getattr(exc, 'status', None)})")
                        return
                    if rate:
                        st["rate_limited"] += 1
                        rl += 1
                    else:
                        tries += 1
                    if rl >= rate_limit_attempts or tries >= attempts:
                        guard.release(worst)
                        failed_bar(name)
                        return
                if not rate:
                    nap(backoff_base * 2 ** (tries - 1))
                continue
            pacer.ok()
            with lock:
                charged = guard.settle(worst, usage.get("input_tokens"), usage.get("output_tokens"), fallback)
                fh.write(json.dumps({"symbol": sym, "bar_ms": bar_ms, "bar_utc": iso(bar_ms), "model": model,
                                     "judgment": raw, "usage": usage,
                                     "usage_estimated": usage.get("input_tokens") is None,
                                     "budget_input_tokens": charged,
                                     "latency_ms": round((time.perf_counter() - t_call) * 1000, 1),
                                     "answered_at": iso(int(time.time() * 1000))}, ensure_ascii=False) + "\n")
                fh.flush()
                st["consecutive"] = 0
                st["new_calls"] += 1
                n = st["new_calls"]
                if n % 100 == 0:
                    os.fsync(fh.fileno())
                if n % 1000 == 0:
                    rate_s = n / max(1e-9, time.monotonic() - st["t0"])
                    left = len(todo) - st["next"]
                    log(f"{n} new calls ({rate_s:.2f}/s, ~{left / max(rate_s, 1e-9) / 3600:.1f} h left), total "
                        f"{guard.calls}, spent ${guard.total_cost():.4f}, projection ${guard.projection() or 0:.4f}, "
                        f"rate-limited {st['rate_limited']}, at {iso(bar_ms)}")
            return

    def worker() -> None:
        cl = client if shared else client()
        try:
            while not stop_evt.is_set():
                with lock:
                    if st["next"] >= len(todo):
                        return
                    bar_ms, sym, j = todo[st["next"]]
                    st["next"] += 1
                cs = candles[sym]
                compact, _payload, body = build_request(sym, tuple(cs[max(0, j + 1 - window): j + 1]), equity)
                fallback = math.ceil(len(body) / 2)  # conservative: the proxy tokenizer gives ~2.2 chars/token
                with lock:
                    while True:
                        if stop_evt.is_set():
                            return
                        worst = guard.worst_case_tokens(fallback)
                        reason = guard.reserve(worst)
                        if reason is None:
                            break
                        if guard.inflight > 0 and guard.stop_reason() is None:
                            # the limit is only reached counting in-flight worst cases: wait until they settle
                            # (a failed one frees its slot), then decide
                            slots.wait(timeout=0.05)
                            continue
                        halt(reason)
                        return
                ask(cl, compact, bar_ms, sym, fallback, worst)
        except Exception as exc:  # noqa: BLE001 - a bug must stop the run, not hang it
            with lock:
                halt(f"worker crashed: {type(exc).__name__}: {exc}")
        finally:
            if not shared:
                close = getattr(cl, "close", None)
                if callable(close):
                    close()

    threads = [threading.Thread(target=worker, name=f"jev-worker-{k}", daemon=True) for k in range(max(1, workers))]
    try:
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
            with lock:
                halt("interrupted (Ctrl-C); rerun the same command to resume")
            log("Ctrl-C: no new calls; waiting for in-flight calls (already paid) to be saved ...")
            for t in threads:
                t.join()
    finally:
        with lock:
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
            eh.close()
    stop = st["stop"]
    summary = {"stop_reason": stop or "plan complete", "new_calls": st["new_calls"], "total_calls": guard.calls,
               "requests_sent": st["requests"], "rate_limited": st["rate_limited"],
               "input_tokens": guard.input_tokens, "output_tokens": guard.output_tokens,
               "cost_usd": round(guard.cost(), 6), "prior_usd": guard.prior_usd,
               "total_cost_usd": round(guard.total_cost(), 6), "projection_usd": guard.projection(),
               "estimated_usage_responses": guard.estimated_usage, "workers": workers}
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
    ap.add_argument("--exclude", default="", help="comma list removed from the universe (recorded in outputs)")
    ap.add_argument("--testnet-missing", default=",".join(TESTNET_MISSING_DEFAULT),
                    help="symbols without testnet markets (reported as a second subset)")
    ap.add_argument("--candle-venue", default="mainnet", choices=("mainnet", "testnet"))
    ap.add_argument("--cache", default="/tmp/jevbt/cache")
    ap.add_argument("--window", type=int, default=300, help="closed bars per state (live buffer max 300)")
    ap.add_argument("--equity", type=float, default=DEFAULT_EQUITY, help="cash_usdt shown in the flat state")
    ap.add_argument("--price-per-mtok", type=float, default=DEFAULT_PRICE_PER_MTOK, help="USD per 1M input tokens")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json path, or 'none'")
    ap.add_argument("--samples", type=int, default=2, help="sample requests per symbol written to the out dir")
    ap.add_argument("--workers", type=int, default=4,
                    help="--live: parallel request threads; dry-run: token-counting processes")
    ap.add_argument("--out", default="/tmp/jevbt/out/month")
    ap.add_argument("--dry-run", action="store_true", default=True, help="default; never calls the model")
    ap.add_argument("--live", action="store_true", help="paid Jev calls (also needs --i-have-approval)")
    ap.add_argument("--i-have-approval", action="store_true")
    ap.add_argument("--max-calls", type=int, default=0, help="hard cap for --live (all answers in the file count)")
    ap.add_argument("--budget-usd", type=float, default=4.0, help="hard cap on real cost from SDK usage")
    ap.add_argument("--check-after", type=int, default=2000, help="calls before the projection check starts")
    ap.add_argument("--answers", default=None, help="append-only answers JSONL (default <out>/live_answers.jsonl)")
    ap.add_argument("--min-interval", type=float, default=0.2, help="seconds between any two call starts (all workers)")
    ap.add_argument("--env-file", default=None,
                    help="--live only: read ONLY TYPESAFE_API_KEY / typesafe_API_KEY from this dotenv file with the "
                         "bot's parser (python-dotenv + jev_trader.config.map_typesafe_api_key); nothing is exported "
                         "or printed. Without it the key must be in the process environment")
    ap.add_argument("--check-ledger", default=None, metavar="LEDGER",
                    help="only compare rebuilt states with logged state_text (read-only ledger copy)")
    ap.add_argument("--check-n", type=int, default=150)
    ap.add_argument("--tag", default="", help="suffix for the dry-run output file names")
    ap.add_argument("--import-from", default=None, metavar="OLD_ANSWERS",
                    help="--live/--import-only: seed a NEW answers file with the in-plan answers of an earlier run "
                         "(old files are only read); its other paid calls count as prior spend against the cap")
    ap.add_argument("--import-only", action="store_true",
                    help="do the --import-from step, write the meta and print the projection; no key, no model call")
    ap.add_argument("--progress", action="store_true",
                    help="only print progress and real spend of --answers (no network, no key)")
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
    excl = exclude_set(args)
    missing = sorted(excl - set(symbols))
    if missing:
        raise SystemExit(f"--exclude names symbols that are not in the universe: {', '.join(missing)}")
    return sorted(s for s in symbols if s not in excl), table


def exclude_set(args: argparse.Namespace) -> set[str]:
    return {s.strip().upper() for s in (getattr(args, "exclude", "") or "").split(",") if s.strip()}


def universe_rule(args: argparse.Namespace) -> str | None:
    parts = []
    if args.universe_ledger:
        parts.append(f"top {args.universe_size} by model answers on {args.universe_days} (UTC, inclusive)")
    if args.symbols:
        parts.append("explicit --symbols list")
    if exclude_set(args):
        parts.append("minus " + ", ".join(sorted(exclude_set(args))) + " (--exclude)")
    return "; ".join(parts) or None


def resolve_api_key(env_file: str | None) -> tuple[str, str]:
    """(key, source) for the live path. Never prints or logs the value.

    --env-file: the bot's own loader semantics (python-dotenv parser, `typesafe_API_KEY` mapped onto
    `TYPESAFE_API_KEY` by jev_trader.config.map_typesafe_api_key, the file wins like load_settings(env_file)),
    but only the key is taken: nothing is put into os.environ. Otherwise the process environment.
    """
    if env_file:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise SystemExit(f"--env-file {path}: no such file; nothing was called")
        from dotenv import dotenv_values

        from jev_trader.config import map_typesafe_api_key

        values = dotenv_values(path)
        sub = {k: str(values.get(k) or "") for k in ("TYPESAFE_API_KEY", "typesafe_API_KEY")}
        key = (map_typesafe_api_key(sub) or "").strip()
        if key:
            return key, f"--env-file {path.name}"
    key = (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("typesafe_API_KEY") or "").strip()
    if key:
        return key, "process environment"
    where = f"neither in --env-file {env_file} nor in" if env_file else "not set in"
    raise SystemExit(f"TYPESAFE_API_KEY (or typesafe_API_KEY) is {where} the process environment; nothing was called")


def progress(answers: Path, price_per_mtok: float) -> dict[str, Any]:
    """Calls done, tokens and real spend so far from the answers JSONL (+ billed errors, meta, status)."""
    _, recs = load_done(answers)
    errs = []
    ep = answers.with_suffix(".errors.jsonl")
    if ep.is_file():
        for line in ep.open(encoding="utf-8"):
            try:
                errs.append(json.loads(line))
            except ValueError:
                continue
    billed = [e for e in errs if e.get("billed")]
    meta_p = answers.with_suffix(".meta.json")
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    planned = meta.get("planned_calls")
    tok = sum(int(r.get("budget_input_tokens") or (r.get("usage") or {}).get("input_tokens") or 0) for r in recs + billed)
    out_tok = sum(int((r.get("usage") or {}).get("output_tokens") or 0) for r in recs + billed)
    calls = len(recs) + len(billed)
    spent = tok / 1e6 * price_per_mtok
    prior = float((meta.get("import") or {}).get("prior_spend_usd") or 0.0)
    recent = [charged_tokens(r) for r in recs[-2000:]]
    per = max(spent / calls, sum(recent) / len(recent) / 1e6 * price_per_mtok) if calls and recent else None
    res = {"answers": len(recs), "billed_unparsable": len(billed), "planned_calls": planned,
           "imported": (meta.get("import") or {}).get("imported_answers", 0),
           "done_pct": round(100 * calls / planned, 2) if planned else None,
           "input_tokens": tok, "output_tokens": out_tok, "input_tokens_per_call": round(tok / calls, 1) if calls else None,
           "spent_in_file_usd": round(spent, 4), "prior_out_of_plan_usd": round(prior, 4),
           "spent_total_usd": round(prior + spent, 4),
           "projected_total_usd": round(prior + spent + per * max(0, planned - calls), 4) if per and planned else None,
           "budget_usd": meta.get("budget_usd"), "rate_limited": sum(1 for e in errs if e.get("rate_limited")),
           "other_errors": sum(1 for e in errs if not e.get("rate_limited") and not e.get("billed")),
           "last_bar_utc": recs[-1].get("bar_utc") if recs else None,
           "last_answer_at": max((r.get("answered_at") or "" for r in recs), default=None)}
    st = answers.parent / "live_status.json"
    if st.is_file():
        res["last_run_stop_reason"] = json.loads(st.read_text()).get("stop_reason")
    return res


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = parse_args(argv)
    if args.progress:
        answers = Path(args.answers) if args.answers else Path(args.out) / "live_answers.jsonl"
        res = progress(answers, args.price_per_mtok)
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return res
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
    if args.live or args.import_only:
        if args.live and (not args.i_have_approval or args.max_calls <= 0 or args.budget_usd <= 0):
            raise SystemExit("--live needs --i-have-approval, --max-calls > 0 and --budget-usd > 0; nothing was called")
        key, key_source = resolve_api_key(args.env_file) if args.live else ("", "")  # before any network access
        answers = Path(args.answers) if args.answers else out / "live_answers.jsonl"
        answers.parent.mkdir(parents=True, exist_ok=True)
        log_path = answers.with_suffix(".log")

        def log(msg: str) -> None:
            line = f"[{datetime.now(IL).strftime('%Y-%m-%d %H:%M:%S')} IL] {msg}"
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(line + "\n")

        meta_path = answers.with_suffix(".meta.json")
        meta = {"symbols": symbols, "universe_rule": universe_rule(args), "range_utc": [iso(start_ms), iso(end_ms)],
                "candle_venue": args.candle_venue, "window": args.window, "equity": args.equity,
                "budget_usd": args.budget_usd, "price_per_mtok": args.price_per_mtok, "max_calls": args.max_calls,
                "check_after": args.check_after, "decisions": list(PRE_RUN_DECISIONS)}
        prev = json.loads(meta_path.read_text()) if meta_path.is_file() else None
        if prev is not None:
            diff = [k for k in ("symbols",) + PLAN_FIXED_KEYS if prev.get(k) != meta[k]]
            if diff:
                raise SystemExit(f"resume refused: {', '.join(diff)} differ from {meta_path.name} (the plan is fixed "
                                 "before the run); nothing was called")
            if args.import_from and not prev.get("import"):
                raise SystemExit("--import-from given but this answers file was started without an import; "
                                 "nothing was called")
            if args.import_from and Path(prev["import"]["source"]).resolve() != Path(args.import_from).resolve():
                raise SystemExit("--import-from differs from the import recorded in the meta; nothing was called")
        candles = {s: load_candles(args.candle_venue, s, fetch_start, end_ms, Path(args.cache)) for s in symbols}
        plan = build_plan(candles, start_ms, end_ms)
        meta["planned_calls"] = len(plan)
        if prev is None:
            if args.import_from:
                meta["import"] = import_prior(Path(args.import_from), answers, {(p[1], p[0]) for p in plan}, meta,
                                              args.price_per_mtok)
                log(f"imported {meta['import']['imported_answers']} answers + "
                    f"{meta['import']['imported_billed_unparsable']} billed-unparsable from {args.import_from}; prior "
                    f"total spend ${meta['import']['prior_total_spend_usd']:.4f}, of it out-of-plan "
                    f"${meta['import']['prior_spend_usd']:.4f}")
            meta["started_at"] = iso(int(time.time() * 1000))
            meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
            prev = meta
        prior_usd = float((prev.get("import") or {}).get("prior_spend_usd") or 0.0)
        guard = BudgetGuard(args.budget_usd, args.price_per_mtok, planned_calls=len(plan), max_calls=args.max_calls,
                            check_after=args.check_after, prior_usd=prior_usd)
        seed_guard(guard, answers)
        if args.import_only:
            res = {"planned_calls": len(plan), "in_file_calls": guard.calls, "prior_out_of_plan_usd": prior_usd,
                   "spent_total_usd": round(guard.total_cost(), 6), "cost_per_call_usd": guard.cost_per_call(),
                   "projection_total_usd": guard.projection(), "budget_usd": args.budget_usd,
                   "headroom_usd": None if guard.projection() is None else round(args.budget_usd - guard.projection(), 4),
                   "import": prev.get("import")}
            log("import-only: " + json.dumps(res))
            return res
        log(f"live month replay: {len(symbols)} symbols {','.join(symbols)}; rule: {meta['universe_rule']}; "
            f"key from {key_source}; workers {args.workers}; answers {answers}; prior out-of-plan spend "
            f"${prior_usd:.4f}; total spent so far ${guard.total_cost():.4f}; projection "
            f"${guard.projection() or 0:.4f} of ${args.budget_usd:.2f}")

        def factory() -> JevUsageClient:
            return JevUsageClient(api_key=key)

        summary = run_live(plan, candles, factory, answers, guard, window=args.window, equity=args.equity,
                           workers=args.workers, min_interval=args.min_interval, log=log)
        del key
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
        "symbols": symbols, "universe_rule": universe_rule(args), "excluded": sorted(exclude_set(args)),
        "decisions": PRE_RUN_DECISIONS,
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
    _res = main()
    # exit 3 when a live run stopped before completing the plan (guard, errors, Ctrl-C); 0 otherwise
    sys.exit(3 if isinstance(_res, dict) and _res.get("stop_reason") not in (None, "plan complete") else 0)
