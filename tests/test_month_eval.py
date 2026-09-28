"""scripts/month_eval.py on a synthetic market (no network)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import backtest_replay as b  # noqa: E402
import month_eval as me  # noqa: E402

BAR = b.BAR_MS
T0 = 1_788_000_000_000 - (1_788_000_000_000 % BAR)


def _market(n_bars: int) -> b.Market:
    bars, atr = {}, {}
    for k, (sym, px) in enumerate((("BTCUSDT", 100_000.0), ("ETHUSDT", 3_000.0))):
        rng = random.Random(k)
        rows, p = [], px
        for i in range(n_bars):
            o = p
            c = o * (1 + rng.gauss(0, 0.002))
            rows.append([T0 + i * BAR, o, max(o, c) * 1.001, min(o, c) * 0.999, c])
            p = c
        bars[sym] = {r[0]: r for r in rows}
        a = b.wilder_atr(rows)
        atr[sym] = {rows[i][0]: a[i] for i in range(len(rows)) if a[i]}
    return b.Market(bars, atr, {"BTCUSDT": [], "ETHUSDT": []}, {"BTCUSDT": 0.1, "ETHUSDT": 0.01}, [])


def _answers(path: Path, start: int, mid: int, end: int) -> None:
    rng = random.Random(7)
    with path.open("w") as fh:
        for t in range(start, end, BAR):
            for sym in ("BTCUSDT", "ETHUSDT"):
                # second-half values are pushed up so a threshold leaking from them would show
                stn = rng.random() * (0.6 if t < mid else 1.0)
                j = {"action": "buy_long" if rng.random() < 0.5 else "hold", "should_trade_now": stn,
                     "trend_aligned": 0.5, "false_break_risk": 0.5, "signal_strength": "слабый", "model": "jev-1.13.0"}
                fh.write(json.dumps({"symbol": sym, "bar_ms": t, "model": "jev-1.13.0", "judgment": j}) + "\n")
            fh.write(json.dumps({"symbol": "BTCUSDT", "bar_ms": t, "model": "laya:x",
                                 "judgment": {**j, "model": "laya:x"}}) + "\n")  # ignored: not Jev
        fh.write('{"symbol": "BTCUSDT", "bar')  # torn line


def test_threshold_from_first_half_only_and_both_halves_evaluated(tmp_path: Path) -> None:
    market = _market(200 + 2 * 288)
    start, end = T0 + 200 * BAR, T0 + (200 + 2 * 288) * BAR
    mid = start + 288 * BAR
    path = tmp_path / "a.jsonl"
    _answers(path, start, mid, end)
    answers = me.load_answers(path)
    assert len(answers) == 2 * 2 * 288
    first = [a["stn"] for a in answers if a["bar_ms"] < mid]
    thr, n = me.threshold(answers, start, mid)
    assert n == len(first) == 576 and abs(thr - b.pct(first, 0.9)) < 1e-12 and thr < 0.6
    res = me.evaluate(answers, market, start, mid, end, seeds=20)
    ins, oos = res["halves"]
    assert ins["half"] == "in-sample" and oos["half"] == "out-of-sample"
    assert ins["signals"] == sum(1 for a in answers if a["bar_ms"] < mid and a["action"] == "buy_long" and a["stn"] >= thr)
    assert oos["signals"] > ins["signals"]  # same threshold, larger second-half values
    for r in (ins, oos):
        assert r["trades"] > 0 and 0 <= r["rank"] <= 100 and r["btc_hold_net"] is not None
        assert r["rand_trades_short"] == 0
    assert "threshold should_trade_now" in me.render(res)
