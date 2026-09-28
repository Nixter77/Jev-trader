#!/usr/bin/env python3
"""Pre-registered evaluation of the paid month replay (answers JSONL from jev_month_replay.py --live).

Fixed before any answer exists; nothing here is tuned on the results.
  answers    only Jev answers (model id starting with "jev"); one per (symbol, 5m state bar S)
  halves     in-sample   [--start, --mid)  = 29.08 00:00 .. 13.09 12:00 UTC
             out-sample  [--mid, --end)    = 13.09 12:00 .. 28.09 00:00 UTC   (by state bar S)
  threshold  90th percentile (linear interpolation) of should_trade_now over ALL Jev answers of
             the in-sample half; applied unchanged to both halves
  entry      action == buy_long and should_trade_now >= threshold
  mechanics  backtest_holds 'trail': maker entry at close(S) alive for bar S+5m only (fills if
             low < limit, maker 0.02 %, fee deducted from cash); stop close(S) - 1.5 x ATR14(S),
             raised after every bar close to max(close since entry) - 1.5 x ATR14 (never lowered);
             no time exit; stop fills at min(open, stop) minus 1 tick, taker 0.05 %; <= 3 positions,
             one per symbol; 0.5 % risk of mark-to-market equity / (1.5 x ATR), margin cap 3x;
             historical funding; start equity 3044.35 for EACH half; positions still open at the
             end of a half are sold at its last close minus 1 tick (taker)
  random     500 seeds, same trade count, same mechanics (backtest_holds.random_nets), slots drawn
             from the same half's Jev answer bars; minus-top-5 removes each seed's own top 5
  BTC hold   BTCUSDT bought at the half's first open, sold at its last close (taker both sides)
Candles/funding: public Binance (mainnet by default: what the model saw), via backtest_replay.
Universe (decided 2026-09-28 before the paid run, see jev_month_replay.PRE_RUN_DECISIONS): top-15 symbols by
model answers 21-23.09 UTC minus SKHYNIXUSDT (fewest answers), 14 symbols, to fit the $4 cap. The report header
repeats these decisions and the universe recorded by the live run (<answers>.meta.json).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_holds as h  # noqa: E402
import backtest_replay as b  # noqa: E402
from jev_month_replay import PRE_RUN_DECISIONS  # noqa: E402

BAR = b.BAR_MS
DEFAULT_START = "2026-08-29T00:00"
DEFAULT_MID = "2026-09-13T12:00"
DEFAULT_END = "2026-09-28T00:00"
EQUITY = 3044.35
QUANTILE = 0.90


def ms(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def iso(t: int) -> str:
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def load_answers(path: Path) -> list[dict[str, Any]]:
    """Jev answers, one per (symbol, bar_ms); torn lines are ignored."""
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            j = rec.get("judgment") if isinstance(rec, dict) else None
            if not isinstance(j, dict):
                continue
            model = str(rec.get("model") or j.get("model") or "")
            if not model.lower().startswith("jev"):
                continue
            try:
                seen.setdefault((str(rec["symbol"]).upper(), int(rec["bar_ms"])), {
                    "symbol": str(rec["symbol"]).upper(), "bar_ms": int(rec["bar_ms"]), "action": str(j["action"]),
                    "stn": float(j["should_trade_now"])})
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(seen.values(), key=lambda r: (r["bar_ms"], r["symbol"]))


def threshold(answers: list[dict[str, Any]], start_ms: int, mid_ms: int) -> tuple[float, int]:
    vals = [a["stn"] for a in answers if start_ms <= a["bar_ms"] < mid_ms]
    return (b.pct(vals, QUANTILE), len(vals)) if vals else (float("nan"), 0)


def evaluate_half(name: str, answers: list[dict[str, Any]], thr: float, market: b.Market, A: h.Arrays,
                  h0: int, h1: int, equity: float, seeds: int) -> dict[str, Any]:
    E = A.idx(h1) - 1  # last bar processed: the one starting 5 min before the half ends
    i0 = A.idx(h0)
    rows = [a for a in answers if h0 <= a["bar_ms"] < h1 and a["symbol"] in market.bars
            and 0 <= A.idx(a["bar_ms"]) <= E and not math.isnan(A.c[a["symbol"]][A.idx(a["bar_ms"])])]
    outs = {(a["symbol"], A.idx(a["bar_ms"])): h.outcome(A, a["symbol"], A.idx(a["bar_ms"]), None, E) for a in rows}
    sig = [a for a in rows if a["action"] == "buy_long" and a["stn"] >= thr]
    buys = sorted((A.idx(a["bar_ms"]) + 1, a["symbol"], A.idx(a["bar_ms"])) for a in sig)
    cnt: dict[str, int] = {}
    trades = h.replay(buys, outs, A, equity, cnt)
    st = h.stats(trades)
    dd, ddp = h.max_dd(h.curve(trades, A, equity, i0, E))
    pool = sorted((o for o in outs.values() if o is not None), key=lambda o: (o.T, o.sym))
    rn, rn5, got, info = h.random_nets(pool, st["trades"], A, equity, seeds) if seeds else ([], [], [], {})
    btc = None
    if "BTCUSDT" in market.bars and market.bar("BTCUSDT", h0) is not None:
        btc = b.summarize(b.hold_btc(market, equity, h0, h1 - BAR), equity)
    return {
        "half": name, "from": iso(h0), "to": iso(h1), "jev_answers": len(rows),
        "buy_long": sum(1 for a in rows if a["action"] == "buy_long"), "signals": len(sig),
        **st, "max_dd": dd, "max_dd_pct": ddp, "counters": cnt,
        "rand_p5": b.pct(rn, 0.05), "rand_median": b.pct(rn, 0.5), "rand_p95": b.pct(rn, 0.95),
        "rank": h.prank(rn, st["net"]),
        "rand5_p5": b.pct(rn5, 0.05), "rand5_median": b.pct(rn5, 0.5), "rand5_p95": b.pct(rn5, 0.95),
        "rank_minus_top5": h.prank(rn5, st["net_minus_top5"]),
        "rand_trades_min": min(got) if got else None, "rand_trades_short": sum(1 for g in got if g < st["trades"]),
        "rand_p_entry": info.get("p"),
        "btc_hold_net": None if btc is None else btc["net"], "btc_hold_max_dd": None if btc is None else btc["max_dd"],
    }


def evaluate(answers: list[dict[str, Any]], market: b.Market, start_ms: int, mid_ms: int, end_ms: int,
             equity: float = EQUITY, seeds: int = 500) -> dict[str, Any]:
    thr, n_thr = threshold(answers, start_ms, mid_ms)
    A = h.build_arrays(market)
    halves = [evaluate_half("in-sample", answers, thr, market, A, start_ms, mid_ms, equity, seeds),
              evaluate_half("out-of-sample", answers, thr, market, A, mid_ms, end_ms, equity, seeds)]
    return {"threshold": thr, "threshold_quantile": QUANTILE, "threshold_from_answers": n_thr,
            "threshold_window": [iso(start_ms), iso(mid_ms)], "equity": equity, "seeds": seeds, "halves": halves}


def render(res: dict[str, Any]) -> str:
    L = [f"# Month evaluation (pre-registered), {res.get('venue', '?')} candles, {res['seeds']} random seeds", ""]
    L += [f"- decided before the run: {d}" for d in res.get("decisions", PRE_RUN_DECISIONS)]
    meta = res.get("run_meta") or {}
    if meta:
        L.append(f"- live run meta: {len(meta.get('symbols', []))} symbols {', '.join(meta.get('symbols', []))}; "
                 f"rule: {meta.get('universe_rule')}; planned calls {meta.get('planned_calls')}; "
                 f"budget ${meta.get('budget_usd')}")
    L.append(f"- symbols with Jev answers in the file ({len(res.get('answer_symbols', []))}): "
             f"{', '.join(res.get('answer_symbols', []))}")
    L += ["",
         f"threshold should_trade_now >= {res['threshold']:.4f} (p{int(res['threshold_quantile'] * 100)} of "
         f"{res['threshold_from_answers']} Jev answers, {res['threshold_window'][0]} .. {res['threshold_window'][1]})", "",
         "| half | Jev answers | buy_long | signals | net | price | fees | funding | trades | win% | avg win | avg loss | "
         "max DD | random p5/med/p95 | rank | net-top5 | rand-top5 p5/med/p95 | rank-top5 | BTC hold |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["halves"]:
        btc = "n/a" if r["btc_hold_net"] is None else f"{r['btc_hold_net']:+.2f}"
        L.append(f"| {r['half']} ({r['from']} .. {r['to']}) | {r['jev_answers']} | {r['buy_long']} | {r['signals']} | "
                 f"{r['net']:+.2f} | {r['price']:+.2f} | {-r['fees']:.2f} | {-r['funding']:+.2f} | {r['trades']} | "
                 f"{r['win_rate'] * 100:.1f} | {r['avg_win']:+.2f} | {r['avg_loss']:+.2f} | "
                 f"{r['max_dd']:.2f} ({r['max_dd_pct'] * 100:.1f}%) | "
                 f"{r['rand_p5']:+.0f}/{r['rand_median']:+.0f}/{r['rand_p95']:+.0f} | {r['rank']:.1f} | "
                 f"{r['net_minus_top5']:+.2f} | {r['rand5_p5']:+.0f}/{r['rand5_median']:+.0f}/{r['rand5_p95']:+.0f} | "
                 f"{r['rank_minus_top5']:.1f} | {btc} |")
    L.append("")
    for r in res["halves"]:
        L.append(f"- {r['half']}: exits {r['exits']}; counters {r['counters']}; random entry p {r['rand_p_entry']}, "
                 f"seeds short of the trade count {r['rand_trades_short']} (min {r['rand_trades_min']})")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", default="/tmp/jevbt/out/month/live_answers.jsonl")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--mid", default=DEFAULT_MID)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--venue", choices=sorted(b.HOSTS), default="mainnet")
    ap.add_argument("--cache", default="/tmp/jevbt/cache")
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--equity", type=float, default=EQUITY)
    ap.add_argument("--out", default="/tmp/jevbt/out/month", help="writes month_eval<tag>.md/.json here")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    answers = load_answers(Path(args.answers))
    if not answers:
        raise SystemExit("no Jev answers in the file")
    start_ms, mid_ms, end_ms = ms(args.start), ms(args.mid), ms(args.end)
    symbols = sorted({a["symbol"] for a in answers} | {"BTCUSDT"})
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    market = b.load_market(args.venue, symbols, start_ms, end_ms - BAR, cache)
    res = evaluate(answers, market, start_ms, mid_ms, end_ms, args.equity, args.seeds)
    res["venue"] = args.venue
    res["answers_file"] = args.answers
    res["symbols"] = symbols
    res["answer_symbols"] = sorted({a["symbol"] for a in answers})
    res["decisions"] = list(PRE_RUN_DECISIONS)
    meta_path = Path(args.answers).with_suffix(".meta.json")
    res["run_meta"] = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"month_eval{args.tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    text = render(res)
    (out / f"month_eval{args.tag}.md").write_text(text)
    print(text)
    return res


if __name__ == "__main__":
    main()
