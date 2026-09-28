#!/usr/bin/env python3
"""Rebuild the bot's Jev requests for a historical range and estimate calls, tokens and cost.

DRY RUN IS THE DEFAULT: no model call is made, no API key is read. The live path
exists for a later, approved run only; it needs BOTH `--live` and
`--i-have-approval`, reads TYPESAFE_API_KEY from the process environment at run
time (never from a file) and stops after --max-calls.

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
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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


# --- reconstruction check against logged states ------------------------------------------

def check_ledger(path: str, venue: str, cache: Path, n: int, seed: int = 2) -> dict[str, Any]:
    """Rebuild logged state_text market lines (2-4) from candles; exact-match rates.

    Reports the match with a fixed 300-bar window and with the best window in 239..300
    (the live buffer length after a restart grows from 239 to 300).
    """
    import random
    import re
    import sqlite3

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = conn.execute("SELECT symbol, state_text FROM decisions WHERE judgment_json IS NOT NULL "
                        "AND state_text LIKE 'symbol=%'").fetchall()
    conn.close()
    recs = []
    for sym, st in rows:
        mm = re.search(r"ts=(\S+)Z", st or "")
        if mm:
            t = int(datetime.strptime(mm.group(1), "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
            recs.append((sym.upper(), t, st))
    t0, t1 = min(r[1] for r in recs), max(r[1] for r in recs)
    sample = random.Random(seed).sample(recs, min(n, len(recs)))
    candles: dict[str, list[Candle]] = {}
    fixed = anyw = total = 0
    for sym, t, st in sample:
        if sym not in candles:
            candles[sym] = load_candles(venue, sym, t0 - 320 * BAR_MS, t1 + BAR_MS, cache)
        cs = candles[sym]
        idx = {c.ts: i for i, c in enumerate(cs)}
        if t not in idx:
            continue
        j, logged = idx[t], st.split("\n")[1:4]
        total += 1
        for w in [300] + list(range(239, 300)):
            compact, _, _ = build_request(sym, tuple(cs[max(0, j + 1 - w): j + 1]), 1.0)
            if compact.as_text().split("\n")[1:4] == logged:
                fixed += w == 300
                anyw += 1
                break
    return {"venue": venue, "checked": total, "match_window_300": fixed / total if total else None,
            "match_some_window_239_300": anyw / total if total else None}


# --- live path (not run without explicit approval) ------------------------------------

def run_live(args: argparse.Namespace, symbols: list[str], start_ms: int, end_ms: int, fetch_start: int) -> None:
    """Paid Jev calls. Guarded by --live AND --i-have-approval; key from the environment only."""
    from jev_trader.jev import JevClient

    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is not set in the process environment")
    out = Path(args.out) / "live_answers.jsonl"
    done = 0
    with JevClient(api_key=key) as client, out.open("a", encoding="utf-8") as fh:
        for sym in symbols:
            candles = load_candles(args.candle_venue, sym, fetch_start, end_ms, Path(args.cache))
            for j, c in enumerate(candles):
                if not (start_ms <= c.ts < end_ms) or j + 1 < 50:
                    continue
                if done >= args.max_calls:
                    print(f"stopped at --max-calls {args.max_calls}")
                    return
                win = tuple(candles[max(0, j + 1 - args.window): j + 1])
                snap = snapshot_from_closed_bars(sym, win, tf="5m", book=None,
                                                 position=Position(side="FLAT", size=0.0, cash_usdt=args.equity))
                compact = build_compact_state(snap, compute_features(snap))
                judgment = client.judge(compact)
                fh.write(json.dumps({"symbol": sym, "bar_ms": c.ts, "judgment": judgment.raw}, ensure_ascii=False) + "\n")
                done += 1


# --- main ---------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", default=None, help="UTC date (exclusive), default: today 00:00 UTC")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--symbols", default="", help="comma list")
    ap.add_argument("--symbols-from-ledger", default=None, help="read-only ledger copy; its answered symbols")
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
    ap.add_argument("--max-calls", type=int, default=0, help="hard cap for --live")
    ap.add_argument("--check-ledger", default=None, metavar="LEDGER",
                    help="only compare rebuilt market lines with logged state_text (read-only ledger copy)")
    ap.add_argument("--check-n", type=int, default=150)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = parse_args(argv)
    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end
           else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0))
    end_ms = int(end.timestamp() * 1000)
    start_ms = int((end - timedelta(days=args.days)).timestamp() * 1000)
    fetch_start = start_ms - args.window * BAR_MS
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.symbols_from_ledger:
        symbols = sorted(set(symbols) | set(symbols_from_ledger(args.symbols_from_ledger)))
    if not symbols:
        raise SystemExit("no symbols (use --symbols or --symbols-from-ledger)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.cache).mkdir(parents=True, exist_ok=True)

    if args.check_ledger:
        res = check_ledger(args.check_ledger, args.candle_venue, Path(args.cache), args.check_n)
        (out / f"state_check_{args.candle_venue}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return res
    if args.live:
        if not args.i_have_approval or args.max_calls <= 0:
            raise SystemExit("--live needs --i-have-approval and --max-calls > 0; nothing was called")
        run_live(args, symbols, start_ms, end_ms, fetch_start)
        return None

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
        "summary": [full, tn],
        "scaled": {s["label"]: [scaled(s, 31), scaled(s, 60)] for s in (full, tn)},
        "per_symbol": [{k: v for k, v in r.items() if k != "samples"} for r in rows],
    }
    samples = [smp for r in rows for smp in r["samples"]]
    (out / "sample_requests.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in samples) + "\n")
    (out / "month_dry_run.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (out / "month_dry_run.md").write_text(render(result))
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
    L.append(f"\nelapsed {r['elapsed_sec']} s; no model calls were made")
    return "\n".join(L)


if __name__ == "__main__":
    main()
