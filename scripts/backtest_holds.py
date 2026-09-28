#!/usr/bin/env python3
"""Model entries + fixed holding time (or ATR stop / trail) vs a random-entry baseline.

Offline only: logged model answers from a read-only ledger copy, public Binance
5m candles and funding (cached by backtest_replay's loaders). No model calls.

Entries (exactly as variant A of backtest_replay):
  follow_jev buy_long answers (policy.apply_policy(follow_jev=True)); maker limit
  at close(S) alive for bar T = S + 5m only, fills if low(T) < limit, maker fee;
  size = 0.5 % of mark-to-market equity / (1.5 x ATR14(S)), capped by free margin
  at 3x; no fee budget; <= 3 positions (open + pending entry orders); one per
  symbol. Model 'close' answers are ignored.

Exit schemes (every scheme keeps the protective ATR stop):
  hold30m .. hold8h  market exit (taker, open(bar) minus 1 tick) at the open of
               the first available bar at or after T + hold, where T is the open
               time of the fill bar. Stop = close(S) - 1.5 x ATR14(S), fixed.
  trail        no time exit. Stop starts at close(S) - 1.5 x ATR14(S); after the
               close of every bar i >= T it is raised to
               max(stop, max(close[T..i]) - 1.5 x ATR14(i)) (never lowered) and
               is active from bar i+1.
  stop         hit if low <= stop; filled at min(open, stop) minus 1 tick, taker.
               On the fill bar the stop fills at stop minus 1 tick (conservative).
               A time exit at the open of a bar is taken before that bar's stop.
  funding      historical rate x mark x qty for funding times inside bars after
               the fill bar (the time-exit bar itself is not charged: exit at open).
  end          positions still open at the global end are sold at the last close
               minus 1 tick, taker.
Equity for sizing is marked to market with every fee paid so far (the entry
maker fee included) and funding accrued; backtest_replay.Sim leaves the entry
fee out of its cash, so sizes can differ from it very slightly.

Slices: all, no_20_21 (decisions from UTC 20-21.09 dropped, the run starts at the
first remaining decision), jev, laya, and leave-one-UTC-day-out. Every slice runs
to the global end so positions opened inside it exit by their own rule.

Random baseline (per scheme and slice, --seeds, default 500): as many trades as
the model variant. Candidates are the slice's decision slots (any answer) whose
maker entry would fill. A chronological pass visits each bar's slots in random
order and enters an eligible slot (symbol flat, < 3 open, sizable; same counting
and sizing code as the model) with probability p, calibrated so a seed yields
slightly more than n trades; each seed stops at n trades (so the last ~3 % of the
period can be thinner than the model's). If n is not reachable even with p = 1
(the model's trades are shorter than random ones), the seed keeps what it got and
the report shows the random trade counts. 'minus top-5' drops the 5 largest trade nets from the
model and, separately, from every random seed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_replay as b  # noqa: E402

BAR = b.BAR_MS
SCHEMES: list[tuple[str, int | None]] = [
    ("hold30m", 6), ("hold1h", 12), ("hold2h", 24), ("hold4h", 48), ("hold8h", 96), ("trail", None),
]
DAY_MS = 86_400_000


# --- per-bar arrays -------------------------------------------------------------

@dataclass
class Arrays:
    g0: int  # open time of grid index 0
    n: int
    o: dict[str, list[float]]
    h: dict[str, list[float]]
    l: dict[str, list[float]]
    c: dict[str, list[float]]
    lc: dict[str, list[float]]  # last valid close (forward filled)
    atr: dict[str, list[float]]
    f: dict[str, list[float]]  # funding per unit qty charged in bar i
    cumf: dict[str, np.ndarray]
    tick: dict[str, float]

    def idx(self, t: int) -> int:
        return (t - self.g0) // BAR


def build_arrays(m: b.Market) -> Arrays:
    g0 = min(min(v) for v in m.bars.values())
    g1 = max(max(v) for v in m.bars.values())
    n = (g1 - g0) // BAR + 1
    nan = float("nan")
    A = Arrays(g0, n, {}, {}, {}, {}, {}, {}, {}, {}, {})
    for sym, bars in m.bars.items():
        o, h, l, c, atr, f = ([nan] * n for _ in range(6))
        for t, r in bars.items():
            i = (t - g0) // BAR
            o[i], h[i], l[i], c[i] = r[1], r[2], r[3], r[4]
        for t, a in m.atr.get(sym, {}).items():
            atr[(t - g0) // BAR] = a
        fz = [0.0] * n
        for ft, rate, mark in m.funding.get(sym, ()):
            i = (ft - g0) // BAR
            if 0 <= i < n:
                px = mark or (o[i] if not math.isnan(o[i]) else 0.0)
                fz[i] += px * rate
        lc, last = [nan] * n, nan
        for i in range(n):
            if not math.isnan(c[i]):
                last = c[i]
            lc[i] = last
        A.o[sym], A.h[sym], A.l[sym], A.c[sym], A.lc[sym], A.atr[sym] = o, h, l, c, lc, atr
        A.f[sym] = fz
        A.cumf[sym] = np.cumsum(np.array(fz))
        A.tick[sym] = m.ticks.get(sym) or 0.0
    return A


def tick_of(A: Arrays, sym: str, px: float) -> float:
    return A.tick.get(sym) or max(px * 1e-5, 1e-8)


# --- per-slot outcome (independent of size and of other positions) ---------------

@dataclass(frozen=True)
class Outcome:
    sym: str
    s: int  # state bar index
    T: int  # fill bar index
    t_out: int  # bar index of the exit (E + 1 for end_of_data)
    entry: float
    exit: float
    fund: float  # funding per unit qty
    reason: str


def outcome(A: Arrays, sym: str, s: int, hold: int | None, E: int) -> Outcome | None:
    o, h, l, c, atrs, f = A.o[sym], A.h[sym], A.l[sym], A.c[sym], A.atr[sym], A.f[sym]
    T = s + 1
    if T > E or math.isnan(c[s]) or math.isnan(atrs[s]) or c[s] <= 0 or math.isnan(l[T]):
        return None
    limit = c[s]
    if not l[T] < limit:
        return None
    stop = limit - b.ATR_MULT * atrs[s]
    fund = 0.0
    hi_close = -math.inf
    for i in range(T, E + 1):
        if math.isnan(o[i]):
            continue
        tk = tick_of(A, sym, o[i])
        if i > T:
            if hold is not None and i >= T + hold:
                return Outcome(sym, s, T, i, limit, o[i] - tk, fund, "time")
            fund += f[i]
        if l[i] <= stop:
            px = min(o[i], stop) if i > T else stop
            return Outcome(sym, s, T, i, limit, px - tk, fund, "stop")
        if hold is None:
            hi_close = max(hi_close, c[i])
            if not math.isnan(atrs[i]):
                stop = max(stop, hi_close - b.ATR_MULT * atrs[i])
    px = A.lc[sym][E]
    return Outcome(sym, s, T, E + 1, limit, px - tick_of(A, sym, px), fund, "end_of_data")


# --- replay with sizing -------------------------------------------------------------

@dataclass
class Tr:
    sym: str
    T: int
    t_out: int
    qty: float
    entry: float
    exit: float
    fee_in: float
    fee_out: float
    fund: float
    reason: str

    @property
    def price(self) -> float:
        return self.qty * (self.exit - self.entry)

    @property
    def net(self) -> float:
        return self.price - self.fee_in - self.fee_out - self.fund


def size_qty(open_: list[Tr], cash: float, T: int, sym: str, s: int, A: Arrays, sim_compat: bool = False) -> float:
    """0.5 % of mark-to-market equity / (1.5 x ATR14(S)), capped by free margin at 3x.

    Equity = realized cash + open positions filled before bar T at close(T-1), minus funding
    accrued and entry fees paid; orders placed at bar T are still pending and do not count.
    Returns 0 when the slot cannot be sized (no ATR / no margin).
    """
    close, atr = A.c[sym][s], A.atr[sym][s]
    if math.isnan(close) or math.isnan(atr) or close <= 0 or atr <= 0:
        return 0.0
    eq, used = cash, 0.0
    for p in open_:
        if p.T >= T:
            continue
        px = A.lc[p.sym][T - 1]
        cf = A.cumf[p.sym]
        eq += p.qty * (px - p.entry) - p.qty * (cf[T - 1] - cf[p.T]) - (0.0 if sim_compat else p.fee_in)
        used += p.qty * px / b.LEVERAGE
    return max(0.0, min(eq * b.RISK_PCT / (b.ATR_MULT * atr), max(0.0, eq - used) * b.LEVERAGE / close))


def make_trade(oc: Outcome, qty: float) -> Tr:
    return Tr(oc.sym, oc.T, oc.t_out, qty, oc.entry, oc.exit, qty * oc.entry * b.MAKER, qty * oc.exit * b.TAKER,
              qty * oc.fund, oc.reason)


def replay(buys: list[tuple[int, str, int]], outs: dict[tuple[str, int], Outcome | None], A: Arrays,
           equity0: float, counters: dict[str, int] | None = None, sim_compat: bool = False) -> list[Tr]:
    """buys: (T, sym, s) sorted by (T, sym). Applies max positions, one per symbol, sizing.

    sim_compat=True reproduces backtest_replay.Sim's equity, which never deducts entry fees (validation only).
    """
    cnt = counters if counters is not None else {}
    cash = equity0
    open_: list[Tr] = []
    done: list[Tr] = []
    cur_T, pending = None, 0
    for T, sym, s in buys:
        if T != cur_T:
            cur_T, pending = T, 0
            still = []
            for p in open_:
                if p.t_out < T:
                    cash += p.net + (p.fee_in if sim_compat else 0.0)
                    done.append(p)
                else:
                    still.append(p)
            open_ = still
        if any(p.sym == sym for p in open_):
            cnt["same_symbol"] = cnt.get("same_symbol", 0) + 1
            continue
        if len(open_) + pending >= b.MAX_POSITIONS:
            cnt["max_positions"] = cnt.get("max_positions", 0) + 1
            continue
        qty = size_qty(open_, cash, T, sym, s, A, sim_compat)
        if qty <= 0:
            continue
        oc = outs.get((sym, s))
        if oc is None:  # the order still blocks a slot for the rest of bar T's decisions
            pending += 1
            cnt["entry_unfilled"] = cnt.get("entry_unfilled", 0) + 1
            continue
        open_.append(make_trade(oc, qty))
    done.extend(open_)
    return done


def curve(trades: list[Tr], A: Arrays, equity0: float, i0: int, E: int) -> np.ndarray:
    n = E - i0 + 1
    cv = np.full(n, equity0, dtype=float)
    for t in trades:
        a = max(t.T, i0)
        z = min(t.t_out, E + 1)  # open over [T, t_out - 1]
        if z > a:
            lc = np.array(A.lc[t.sym][a:z])
            cf = A.cumf[t.sym]
            cv[a - i0:z - i0] += t.qty * (lc - t.entry) - t.qty * (cf[a:z] - cf[t.T]) - t.fee_in
        if t.t_out <= E:
            cv[t.t_out - i0:] += t.net
    return cv


def max_dd(cv: np.ndarray) -> tuple[float, float]:
    peak = np.maximum.accumulate(cv)
    dd = peak - cv
    k = int(np.argmax(dd))
    return float(dd[k]), float(dd[k] / peak[k]) if peak[k] else 0.0


def stats(trades: list[Tr]) -> dict[str, Any]:
    nets = [t.net for t in trades]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    holds = [(t.t_out - t.T) * 5 for t in trades if t.reason != "end_of_data"]
    top5 = sorted(nets, reverse=True)[:5]
    return {
        "net": sum(nets), "price": sum(t.price for t in trades), "fees": sum(t.fee_in + t.fee_out for t in trades),
        "funding": sum(t.fund for t in trades), "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "avg_win": statistics.mean(wins) if wins else 0.0, "avg_loss": statistics.mean(losses) if losses else 0.0,
        "top5": sum(top5), "net_minus_top5": sum(nets) - sum(top5), "exits": reasons,
        "median_hold_min": statistics.median(holds) if holds else None,
    }


def minus_top5(nets: list[float]) -> float:
    return sum(nets) - sum(sorted(nets, reverse=True)[:5])


def prank(values: list[float], x: float) -> float:
    lo = sum(1 for v in values if v < x)
    eq = sum(1 for v in values if v == x)
    return 100.0 * (lo + 0.5 * eq) / len(values) if values else float("nan")


# --- random baseline ------------------------------------------------------------------

def random_run(groups: list[tuple[int, list[Outcome]]], p: float, rng: random.Random, A: Arrays, equity0: float,
               n_max: int | None = None) -> list[Tr]:
    """Chronological random entries with the model's mechanics.

    groups: fillable decision slots grouped by entry bar T (ascending). At every bar the slots are
    visited in a random order; an eligible slot (symbol flat, fewer than MAX_POSITIONS open, sizable)
    is entered with probability p. Sizing, counting and settlement are the ones replay() uses.
    Stops after n_max entries.
    """
    cash = equity0
    open_: list[Tr] = []
    done: list[Tr] = []
    total = 0
    for T, g in groups:
        if open_:
            still = []
            for q in open_:
                if q.t_out < T:
                    cash += q.net
                    done.append(q)
                else:
                    still.append(q)
            open_ = still
        if len(open_) >= b.MAX_POSITIONS:
            continue
        order = g[:] if len(g) > 1 else g
        if len(order) > 1:
            rng.shuffle(order)
        for oc in order:
            if len(open_) >= b.MAX_POSITIONS:
                break
            if any(q.sym == oc.sym for q in open_):
                continue
            if p < 1.0 and rng.random() >= p:
                continue
            qty = size_qty(open_, cash, T, oc.sym, oc.s, A)
            if qty <= 0:
                continue
            open_.append(make_trade(oc, qty))
            total += 1
            if n_max is not None and total >= n_max:
                return done + open_
    return done + open_


def group_pool(pool: list[Outcome]) -> list[tuple[int, list[Outcome]]]:
    by: dict[int, list[Outcome]] = {}
    for oc in pool:
        by.setdefault(oc.T, []).append(oc)
    return [(T, sorted(by[T], key=lambda o: o.sym)) for T in sorted(by)]


def calibrate_p(groups, n: int, A: Arrays, equity0: float, probe_seeds: int = 4, over: float = 1.03) -> float:
    """Entry probability whose mean trade count is about n * over (bisection; 1.0 if unreachable)."""
    def mean_count(p: float) -> float:
        return statistics.mean(len(random_run(groups, p, random.Random(10_000 + k), A, equity0))
                               for k in range(probe_seeds))

    if mean_count(1.0) <= n * over:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if mean_count(mid) < n * over:
            lo = mid
        else:
            hi = mid
    return hi


def random_nets(pool: list[Outcome], n: int, A: Arrays, equity0: float, seeds: int
                ) -> tuple[list[float], list[float], list[int], dict[str, Any]]:
    """Per seed: random_run with the calibrated p (raised by 10 % and rerun while the seed has fewer
    than n trades, up to 10 times), keeping its first n entries."""
    groups = group_pool(pool)
    p0 = calibrate_p(groups, n, A, equity0) if n else 0.0
    nets, nets5, got = [], [], []
    retries = 0
    for seed in range(seeds):
        rng = random.Random(seed)
        p = p0
        trades = random_run(groups, p, rng, A, equity0, n_max=n)
        tries = 0
        while len(trades) < n and p < 1.0 and tries < 10:
            p = min(1.0, p * 1.1)
            trades = random_run(groups, p, rng, A, equity0, n_max=n)
            tries += 1
        retries += tries > 0
        tn = [t.net for t in trades]
        nets.append(sum(tn))
        nets5.append(minus_top5(tn))
        got.append(len(trades))
    return nets, nets5, got, {"p": p0, "seeds_retried": retries, "trades_median": statistics.median(got) if got else None}


# --- driver ------------------------------------------------------------------------------

def utc_day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%m-%d")


def slices_of(decs: list[b.Decision]) -> dict[str, list[b.Decision]]:
    days = sorted({utc_day(d.bar_ms) for d in decs})
    out = {
        "all": decs,
        "no_20_21": [d for d in decs if utc_day(d.bar_ms) not in ("09-20", "09-21")],
        "jev": [d for d in decs if not d.model.lower().startswith("laya")],
        "laya": [d for d in decs if d.model.lower().startswith("laya")],
    }
    for day in days:
        out[f"loo_{day}"] = [d for d in decs if utc_day(d.bar_ms) != day]
    return out


def run_venue(venue: str, ledger: str, cache: Path, seeds: int, only: set[str] | None, log) -> dict[str, Any]:
    decisions, logged_eq = b.load_decisions(ledger)
    equity0 = logged_eq or 10000.0
    t0 = decisions[0].bar_ms + BAR
    t1 = decisions[-1].bar_ms + BAR
    symbols = sorted({d.symbol for d in decisions} | {"BTCUSDT"})
    market = b.load_market(venue, symbols, t0, t1, cache)
    A = build_arrays(market)
    decs = [d for d in decisions if d.symbol in market.bars and not math.isnan(A.c[d.symbol][A.idx(d.bar_ms)])]
    E = A.idx(t1)
    syms = sorted(market.bars)
    buy_keys = {(d.symbol, A.idx(d.bar_ms)) for d in decs
                if b.apply_policy(d.judgment, follow_jev=True).passed and d.judgment.action == "buy_long"}
    t_start = time.time()
    outs_by: dict[str, dict[tuple[str, int], Outcome | None]] = {}
    for name, hold in SCHEMES:
        outs_by[name] = {(d.symbol, A.idx(d.bar_ms)): outcome(A, d.symbol, A.idx(d.bar_ms), hold, E) for d in decs}
    log(f"[{venue}] outcomes for {len(decs)} slots x {len(SCHEMES)} schemes in {time.time() - t_start:.1f}s")
    sl = slices_of(decs)
    res: dict[str, Any] = {
        "meta": {
            "venue": venue, "start_equity": equity0, "period_utc": [iso(t0), iso(t1)],
            "answers_used": len(decs), "answers_total": len(decisions), "follow_jev_buy_slots": len(buy_keys),
            "symbols_without_candles": sorted({d.symbol for d in decisions} - set(market.bars)),
            "funding_missing": market.funding_missing, "seeds": seeds,
            "slices": {k: {"answers": len(v), "start_utc": iso(v[0].bar_ms + BAR) if v else None} for k, v in sl.items()},
        },
        "rows": [],
    }
    for sname, sdecs in sl.items():
        if only and sname not in only:
            continue
        if not sdecs:
            continue
        i0 = A.idx(sdecs[0].bar_ms + BAR)
        buys = sorted((A.idx(d.bar_ms) + 1, d.symbol, A.idx(d.bar_ms)) for d in sdecs
                      if (d.symbol, A.idx(d.bar_ms)) in buy_keys)
        for name, _hold in SCHEMES:
            ts = time.time()
            outs = outs_by[name]
            cnt: dict[str, int] = {}
            trades = replay(buys, outs, A, equity0, cnt)
            st = stats(trades)
            dd, ddp = max_dd(curve(trades, A, equity0, i0, E))
            pool = [outs[(d.symbol, A.idx(d.bar_ms))] for d in sdecs if outs[(d.symbol, A.idx(d.bar_ms))] is not None]
            rn, rn5, got, rinfo = random_nets(pool, st["trades"], A, equity0, seeds) if seeds else ([], [], [], {})
            row = {
                "slice": sname, "scheme": name, **st, "max_dd": dd, "max_dd_pct": ddp, "counters": cnt,
                "rand_p5": b.pct(rn, 0.05), "rand_median": b.pct(rn, 0.5), "rand_p95": b.pct(rn, 0.95),
                "rank": prank(rn, st["net"]),
                "rand5_p5": b.pct(rn5, 0.05), "rand5_median": b.pct(rn5, 0.5), "rand5_p95": b.pct(rn5, 0.95),
                "rank_minus_top5": prank(rn5, st["net_minus_top5"]),
                "rand_trades_short": sum(1 for g in got if g < st["trades"]),
                "rand_trades_min": min(got) if got else None,
                "rand_p_entry": rinfo.get("p"), "rand_seeds_retried": rinfo.get("seeds_retried"),
                "rand_trades_median": rinfo.get("trades_median"),
            }
            res["rows"].append(row)
            log(f"[{venue}] {sname:10s} {name:8s} net {st['net']:+9.2f} n={st['trades']:3d} "
                f"rand p5/med/p95 {row['rand_p5']:+.0f}/{row['rand_median']:+.0f}/{row['rand_p95']:+.0f} "
                f"rank {row['rank']:.1f} | -top5 {st['net_minus_top5']:+.2f} rank {row['rank_minus_top5']:.1f} "
                f"({time.time() - ts:.1f}s)")
    return res


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def validate(ledger: str, venue: str, cache: Path, hold_bars: int) -> tuple[float, int, float, int]:
    """Cross-check the fast engine against backtest_replay.Sim with a time exit bolted on."""
    decisions, eq = b.load_decisions(ledger)
    t0, t1 = decisions[0].bar_ms + BAR, decisions[-1].bar_ms + BAR
    market = b.load_market(venue, sorted({d.symbol for d in decisions} | {"BTCUSDT"}), t0, t1, cache)
    decs = [d for d in decisions if d.symbol in market.bars]

    class HoldSim(b.Sim):
        def on_decision(self, d, t):  # noqa: ANN001
            if d.judgment.action == "close":
                return
            super().on_decision(d, t)

        def on_bar(self, sym, t):  # noqa: ANN001
            p = self.pos.get(sym)
            if p is not None and t >= p.t_in + hold_bars * BAR and self.m.bar(sym, t) is not None:
                self.market_exits[sym] = "time"
            super().on_bar(sym, t)

    if True:
        sim = HoldSim(b.old_rules("A"), market, eq)
        by_bar: dict[int, list] = {}
        for d in decs:
            by_bar.setdefault(d.bar_ms + BAR, []).append(d)
        t = t0
        while t <= t1:
            for d in by_bar.get(t, ()):
                sim.on_decision(d, t)
            for sym in sorted(set(sim.pos) | set(sim.entry_orders) | set(sim.market_exits)):
                sim.on_bar(sym, t)
            t += BAR
        for sym in list(sim.pos):
            px = sim.last_px.get(sym, sim.pos[sym].entry)
            sim.close_pos(sym, t, px - market.tick(sym, px), b.TAKER, "end_of_data")
        sim_trades = sim.res.trades
    A = build_arrays(market)
    E = A.idx(t1)
    decs2 = [d for d in decs if not math.isnan(A.c[d.symbol][A.idx(d.bar_ms)])]
    outs = {(d.symbol, A.idx(d.bar_ms)): outcome(A, d.symbol, A.idx(d.bar_ms), hold_bars, E) for d in decs2}
    buys = sorted((A.idx(d.bar_ms) + 1, d.symbol, A.idx(d.bar_ms)) for d in decs2
                  if b.apply_policy(d.judgment, follow_jev=True).passed and d.judgment.action == "buy_long")
    fast = replay(buys, outs, A, eq, sim_compat=True)
    return sum(t.net for t in sim_trades), len(sim_trades), sum(t.net for t in fast), len(fast)


# --- report ------------------------------------------------------------------------------

def md(res: dict[str, Any]) -> str:
    L = [f"# Hold-time backtest, {res['meta']['venue']}: {res['meta']['period_utc'][0]} .. {res['meta']['period_utc'][1]}, "
         f"equity {res['meta']['start_equity']:.2f}, {res['meta']['seeds']} random seeds", ""]
    by_slice: dict[str, list[dict]] = {}
    for r in res["rows"]:
        by_slice.setdefault(r["slice"], []).append(r)
    for sname, rows in by_slice.items():
        meta = res["meta"]["slices"][sname]
        L += [f"## {sname} ({meta['answers']} answers, start {meta['start_utc']})", "",
              "| scheme | net | price | fees | funding | trades | win% | avg win | avg loss | max DD | med hold m | "
              "rand p5/med/p95 | rank | net-top5 | rand-top5 p5/med/p95 | rank-top5 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(
                f"| {r['scheme']} | {r['net']:+.2f} | {r['price']:+.2f} | {-r['fees']:.2f} | {-r['funding']:+.2f} | "
                f"{r['trades']} | {r['win_rate'] * 100:.1f} | {r['avg_win']:+.2f} | {r['avg_loss']:+.2f} | "
                f"{r['max_dd']:.2f} ({r['max_dd_pct'] * 100:.1f}%) | {'n/a' if r['median_hold_min'] is None else format(r['median_hold_min'], '.0f')} | "
                f"{r['rand_p5']:+.0f}/{r['rand_median']:+.0f}/{r['rand_p95']:+.0f} | {r['rank']:.1f} | "
                f"{r['net_minus_top5']:+.2f} | {r['rand5_p5']:+.0f}/{r['rand5_median']:+.0f}/{r['rand5_p95']:+.0f} | "
                f"{r['rank_minus_top5']:.1f} |")
        L.append("")
    loo = [r for r in res["rows"] if r["slice"].startswith("loo_")]
    if loo:
        L += ["## leave-one-day-out summary (over dropped days)", "",
              "| scheme | net min | net median | net max | rank min | rank median | rank-top5 min | rank-top5 median |",
              "|---|---|---|---|---|---|---|---|"]
        for name, _ in SCHEMES:
            rs = [r for r in loo if r["scheme"] == name]
            nets = [r["net"] for r in rs]
            ranks = [r["rank"] for r in rs]
            r5 = [r["rank_minus_top5"] for r in rs]
            L.append(f"| {name} | {min(nets):+.2f} | {statistics.median(nets):+.2f} | {max(nets):+.2f} | "
                     f"{min(ranks):.1f} | {statistics.median(ranks):.1f} | {min(r5):.1f} | {statistics.median(r5):.1f} |")
        L.append("")
    L.append("exits / guards:")
    for r in res["rows"]:
        if r["slice"] in ("all", "no_20_21", "jev", "laya"):
            L.append(f"- {r['slice']} {r['scheme']}: exits {r['exits']}; counters {r['counters']}; "
                     f"random: entry p {r['rand_p_entry']:.3f}, seeds short of trade count {r['rand_trades_short']} "
                     f"(min {r['rand_trades_min']}, median {r['rand_trades_median']}), seeds needing a higher p {r['rand_seeds_retried']}")
    L.append(f"\nmeta: {json.dumps({k: v for k, v in res['meta'].items() if k != 'slices'}, ensure_ascii=False)}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True, help="copy of the ledger sqlite (opened read-only)")
    ap.add_argument("--venue", choices=sorted(b.HOSTS), default="testnet")
    ap.add_argument("--cache", default="/tmp/jevbt/cache")
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--slices", default="", help="comma list to run a subset (default: all)")
    ap.add_argument("--out", default="/tmp/jevbt/out", help="directory for holds_<venue>.md/.json")
    ap.add_argument("--validate", type=int, default=0, metavar="BARS",
                    help="only cross-check the engine against backtest_replay.Sim for a BARS time exit")
    args = ap.parse_args()
    cache = Path(args.cache)
    if args.validate:
        print("Sim net %.2f n=%d | fast net %.2f n=%d" % validate(args.ledger, args.venue, cache, args.validate))
        return
    only = {s for s in args.slices.split(",") if s} or None
    res = run_venue(args.venue, args.ledger, cache, args.seeds, only, lambda s: print(s, file=sys.stderr, flush=True))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = md(res)
    (out / f"holds_{args.venue}.md").write_text(text)
    (out / f"holds_{args.venue}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print(text)


if __name__ == "__main__":
    main()
