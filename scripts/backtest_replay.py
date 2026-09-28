#!/usr/bin/env python3
"""Offline replay of logged model answers on historical Binance USD-M candles.

No model calls and no signed API calls. The ledger is opened read-only
(sqlite URI mode=ro); pass a copy made with `sqlite3 ... ".backup"`.

Variants
  A   old rules: follow_jev, maker entry, market (taker) exit, no entry caps,
      no window, no loss-streak pause, no daily_loss, no fee-aware sizing.
  A+dl  A with the daily_loss stop (it was active for part of the real week).
  B   new rules: policy gates with should_trade_now >= threshold, maker entry,
      post-only (maker) model exit, fee budget in size, <= 3 entries per UTC day,
      <= 2 per hour, no entries 03-09 Asia/Jerusalem, 3-loss 120 min pause,
      daily_loss -2.5 % (market close), reentry cooldown 30 min, min hold 15 min.
  B@0.45 / B@0.40  informational: B with a lower bar (the logged buy answers
      top out near 0.52, so 0.60 and above never trade).
  B-mech  B's mechanics (maker exit, fee sizing, caps, window, pause, daily_loss)
      on A's follow_jev entries.
  C   random entries: as many trades as the base variant, random decision
      slots, the base's sizing and exit style (ATR stop, time exit after a
      holding time drawn from the base's trades; maker for B, market for A).
  D   hold BTCUSDT the whole period at 1x equity.

Fill model (5m bars, bar T is the one after the decision's state bar S):
  maker entry  limit = close(S), live for bar T only (entry_max_age 300 s),
               fills if low(T) < limit.
  maker exit   limit = close(S), live for bar T only; if it does not fill the
               position stays until the next close answer or the stop.
  market       open(T) minus 1 tick (sell) / plus 1 tick (buy), taker fee.
  stop         1.5 x ATR14(5m) below the decision close, fixed at entry; hit if
               low <= stop, filled at min(open, stop) minus 1 tick, taker fee.
               A position that fills and touches its stop in the same bar is
               stopped out (conservative).
  funding      historical fundingRate x qty x markPrice at each funding time
               the position was open over.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_trader.jev import judgment_from_dict  # noqa: E402
from jev_trader.policy import apply_policy, is_laya_model  # noqa: E402
from jev_trader.risk import (  # noqa: E402
    DEFAULT_LOSS_STREAK_PAUSE_MIN,
    DEFAULT_LOSS_STREAK_PAUSE_N,
    DEFAULT_MAX_ENTRIES_PER_DAY,
    DEFAULT_MAX_ENTRIES_PER_HOUR,
    MIN_HOLD_SEC,
    REENTRY_COOLDOWN_SEC,
    qty_after_fee_budget,
)

BAR_MS = 300_000
TAKER = 0.0005
MAKER = 0.0002
RISK_PCT = 0.005
ATR_MULT = 1.5
LEVERAGE = 3.0
MAX_POSITIONS = 3
DAILY_LOSS = 0.025
IL = ZoneInfo("Asia/Jerusalem")
HOSTS = {"mainnet": "https://fapi.binance.com", "testnet": "https://testnet.binancefuture.com"}


# --- data ---------------------------------------------------------------------

def _get(url: str) -> Any:
    """Public GET via curl (uses the system trust store, unlike bare urllib here)."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            out = subprocess.run(["curl", "-sfS", "-m", "20", url], capture_output=True, check=True, timeout=30)
            return json.loads(out.stdout)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"GET {url}: {last}")


def _cached(cache: Path, name: str, fetch) -> Any:
    path = cache / name
    if path.is_file():
        return json.loads(path.read_text())
    data = fetch()
    path.write_text(json.dumps(data))
    return data


def load_klines(venue: str, symbol: str, start_ms: int, end_ms: int, cache: Path) -> list[list[float]]:
    def fetch() -> list:
        out: list = []
        cur = start_ms
        while cur < end_ms:
            rows = _get(f"{HOSTS[venue]}/fapi/v1/klines?symbol={symbol}&interval=5m&startTime={cur}&endTime={end_ms}&limit=1500")
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
        rows = _cached(cache, f"k_{venue}_{symbol}_{start_ms}_{end_ms}.json", fetch)
    except Exception as exc:  # noqa: BLE001
        print(f"klines {venue} {symbol}: {exc}", file=sys.stderr)
        return []
    return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in rows]


def load_funding(venue: str, symbol: str, start_ms: int, end_ms: int, cache: Path) -> list[tuple[int, float, float]] | None:
    def fetch() -> list:
        return _get(f"{HOSTS[venue]}/fapi/v1/fundingRate?symbol={symbol}&startTime={start_ms}&endTime={end_ms}&limit=1000") or []

    try:
        rows = _cached(cache, f"f_{venue}_{symbol}_{start_ms}_{end_ms}.json", fetch)
    except Exception as exc:  # noqa: BLE001
        print(f"funding {venue} {symbol}: {exc}", file=sys.stderr)
        return None
    if not isinstance(rows, list):
        return None
    return [(int(r["fundingTime"]), float(r["fundingRate"]), float(r.get("markPrice") or 0.0)) for r in rows]


def load_ticks(venue: str, cache: Path) -> dict[str, float]:
    try:
        info = _cached(cache, f"exinfo_{venue}.json", lambda: _get(f"{HOSTS[venue]}/fapi/v1/exchangeInfo"))
    except Exception as exc:  # noqa: BLE001
        print(f"exchangeInfo {venue}: {exc}", file=sys.stderr)
        return {}
    out: dict[str, float] = {}
    for s in info.get("symbols", []):
        for f in s.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                out[s["symbol"]] = float(f["tickSize"])
    return out


def wilder_atr(bars: list[list[float]], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(bars)
    if len(bars) < period + 1:
        return out
    trs = []
    prev = bars[0][4]
    for b in bars:
        trs.append(max(b[2] - b[3], abs(b[2] - prev), abs(b[3] - prev)))
        prev = b[4]
    acc = sum(trs[:period])
    out[period - 1] = acc / period
    for i in range(period, len(trs)):
        acc = acc - acc / period + trs[i]
        out[i] = acc / period
    return out


@dataclass
class Decision:
    bar_ms: int  # open time of the state bar S
    symbol: str
    judgment: Any
    model: str


def load_decisions(ledger_path: str) -> tuple[list[Decision], float | None]:
    uri = f"file:{ledger_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    rows = conn.execute(
        "SELECT id, symbol, judgment_json, state_text FROM decisions WHERE judgment_json IS NOT NULL ORDER BY id"
    ).fetchall()
    conn.close()
    seen: dict[tuple[str, int], Decision] = {}
    start_equity = None
    for _id, symbol, jj, state in rows:
        try:
            data = json.loads(jj)
        except (TypeError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("model") in (None, "model_skipped"):
            continue
        m = re.search(r"\bts=(\d{4}-\d\d-\d\dT\d\d:\d\d)Z", state or "")
        if not m:
            continue
        cash = re.search(r"cash_usdt=([0-9.]+)", state or "")
        if start_equity is None and cash and float(cash.group(1)) != 10000.0:
            start_equity = float(cash.group(1))
        try:
            judgment = judgment_from_dict(data)
        except (KeyError, TypeError, ValueError):
            continue
        bar = int(datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
        seen[(symbol.upper(), bar)] = Decision(bar, symbol.upper(), judgment, str(data.get("model")))
    return sorted(seen.values(), key=lambda d: (d.bar_ms, d.symbol)), start_equity


@dataclass
class Market:
    bars: dict[str, dict[int, list[float]]]
    atr: dict[str, dict[int, float]]
    funding: dict[str, list[tuple[int, float, float]]]
    ticks: dict[str, float]
    funding_missing: list[str]

    def bar(self, sym: str, t: int) -> list[float] | None:
        return self.bars.get(sym, {}).get(t)

    def tick(self, sym: str, price: float) -> float:
        return self.ticks.get(sym) or max(price * 1e-5, 1e-8)


def load_market(venue: str, symbols: list[str], start_ms: int, end_ms: int, cache: Path) -> Market:
    ticks = load_ticks(venue, cache)
    bars: dict[str, dict[int, list[float]]] = {}
    atr: dict[str, dict[int, float]] = {}
    funding: dict[str, list[tuple[int, float, float]]] = {}
    missing: list[str] = []
    for sym in symbols:
        rows = load_klines(venue, sym, start_ms - 200 * BAR_MS, end_ms + 2 * BAR_MS, cache)
        if not rows:
            continue
        bars[sym] = {r[0]: r for r in rows}
        a = wilder_atr(rows)
        atr[sym] = {rows[i][0]: a[i] for i in range(len(rows)) if a[i]}
        f = load_funding(venue, sym, start_ms, end_ms + BAR_MS, cache)
        if f is None:
            missing.append(sym)
            f = []
        funding[sym] = f
    return Market(bars, atr, funding, ticks, missing)


# --- simulation ---------------------------------------------------------------

@dataclass
class Rules:
    name: str
    follow_jev: bool = False
    threshold: float = 0.72
    maker_exit: bool = True
    fee_budget: bool = True
    daily_cap: int = DEFAULT_MAX_ENTRIES_PER_DAY
    hourly_cap: int = DEFAULT_MAX_ENTRIES_PER_HOUR
    window: tuple[int, int] | None = (3, 9)
    loss_streak_n: int = DEFAULT_LOSS_STREAK_PAUSE_N
    loss_streak_min: float = DEFAULT_LOSS_STREAK_PAUSE_MIN
    daily_loss: bool = True
    cooldown_sec: float = REENTRY_COOLDOWN_SEC
    min_hold_sec: float = MIN_HOLD_SEC
    models: str = "all"  # all | jev | laya


def old_rules(name: str = "A", **kw: Any) -> Rules:
    base = dict(follow_jev=True, maker_exit=False, fee_budget=False, daily_cap=0, hourly_cap=0, window=None,
                loss_streak_n=0, daily_loss=False, cooldown_sec=0.0, min_hold_sec=0.0)
    base.update(kw)
    return Rules(name=name, **base)


@dataclass
class Pos:
    qty: float
    entry: float
    stop: float
    t_in: int
    fee: float
    funding: float = 0.0
    reason_in: str = ""


@dataclass
class Trade:
    symbol: str
    t_in: int
    t_out: int
    qty: float
    entry: float
    exit: float
    price_pnl: float
    fees: float
    funding: float
    reason: str

    @property
    def net(self) -> float:
        return self.price_pnl - self.fees - self.funding


@dataclass
class Result:
    name: str
    trades: list[Trade] = field(default_factory=list)
    curve: list[float] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1


class Sim:
    def __init__(self, rules: Rules, market: Market, equity: float) -> None:
        self.r = rules
        self.m = market
        self.cash = equity  # realized equity
        self.pos: dict[str, Pos] = {}
        self.entry_orders: dict[str, dict[str, Any]] = {}
        self.exit_orders: dict[str, dict[str, Any]] = {}
        self.market_exits: dict[str, str] = {}
        self.entries: list[int] = []  # fill times
        self.last_close: dict[str, int] = {}
        self.closes: list[tuple[int, float]] = []
        self.day: int | None = None
        self.day_start = equity
        self.day_blocked = False
        self.last_px: dict[str, float] = {}
        self.res = Result(rules.name)

    # equity -------------------------------------------------------------
    def mtm(self) -> float:
        eq = self.cash
        for sym, p in self.pos.items():
            eq += p.qty * (self.last_px.get(sym, p.entry) - p.entry) - p.funding
        return eq

    def used_margin(self) -> float:
        return sum(p.qty * self.last_px.get(s, p.entry) for s, p in self.pos.items()) / LEVERAGE

    # guards ---------------------------------------------------------------
    def entry_skip(self, t: int) -> str | None:
        r = self.r
        if r.window is not None:
            hour = datetime.fromtimestamp(t / 1000, tz=timezone.utc).astimezone(IL).hour
            if r.window[0] <= hour < r.window[1]:
                return "no_entry_window"
        pending = len(self.entry_orders)
        if r.hourly_cap and sum(1 for x in self.entries if x >= t - 3_600_000) + pending >= r.hourly_cap:
            return "hourly_entry_cap"
        day0 = t - t % 86_400_000
        if r.daily_cap and sum(1 for x in self.entries if x >= day0) + pending >= r.daily_cap:
            return "daily_entry_cap"
        if r.loss_streak_n:
            streak, last = 0, None
            for ts, pnl in reversed(self.closes):
                if pnl < 0:
                    streak += 1
                    last = last or ts
                else:
                    break
            if streak >= r.loss_streak_n and last is not None and t < last + r.loss_streak_min * 60_000:
                return "loss_streak_pause"
        return None

    # decisions at the start of bar t (state bar s = t - 5m) -----------------
    def on_decision(self, d: Decision, t: int) -> None:
        r = self.r
        sym = d.symbol
        if r.models == "jev" and is_laya_model(d.judgment):
            return
        if r.models == "laya" and not is_laya_model(d.judgment):
            return
        s_bar = self.m.bar(sym, d.bar_ms)
        if s_bar is None:
            self.res.bump("no_candle")
            return
        close = s_bar[4]
        if r.daily_loss and self.day_blocked:
            if sym in self.pos:
                self.market_exits[sym] = "daily_loss"
            return
        pol = apply_policy(d.judgment, follow_jev=r.follow_jev, min_should_trade=r.threshold)
        if not pol.passed:
            return
        if pol.action == "close":
            if sym not in self.pos or sym in self.exit_orders or sym in self.market_exits:
                return
            if r.min_hold_sec and (t - self.pos[sym].t_in) < r.min_hold_sec * 1000:
                self.res.bump("min_hold")
                return
            if r.maker_exit:
                self.exit_orders[sym] = {"limit": close, "t": t}
            else:
                self.market_exits[sym] = "model_close"
            return
        if pol.action != "buy_long" or sym in self.pos or sym in self.entry_orders:
            return
        if r.cooldown_sec and sym in self.last_close and t - self.last_close[sym] < r.cooldown_sec * 1000:
            return
        if len(self.pos) + len(self.entry_orders) >= MAX_POSITIONS:
            self.res.bump("max_positions")
            return
        why = self.entry_skip(t)
        if why:
            self.res.bump(why)
            return
        atr = self.m.atr.get(sym, {}).get(d.bar_ms)
        if not atr or close <= 0:
            return
        equity = self.mtm()
        stop_dist = ATR_MULT * atr
        risk = equity * RISK_PCT
        qty = risk / stop_dist
        avail = max(0.0, equity - self.used_margin())
        qty = min(qty, avail * LEVERAGE / close)
        if r.fee_budget:
            qty = qty_after_fee_budget(qty=qty, price=close, risk_amount=risk)
        if qty <= 0:
            return
        self.entry_orders[sym] = {"limit": close, "qty": qty, "stop": close - stop_dist, "t": t}

    # bar processing ---------------------------------------------------------
    def close_pos(self, sym: str, t: int, px: float, fee_rate: float, reason: str) -> None:
        p = self.pos.pop(sym)
        fee_out = p.qty * px * fee_rate
        trade = Trade(sym, p.t_in, t, p.qty, p.entry, px, p.qty * (px - p.entry), p.fee + fee_out, p.funding, reason)
        self.cash += trade.price_pnl - fee_out - p.funding
        self.res.trades.append(trade)
        self.last_close[sym] = t
        self.closes.append((t, trade.net))
        self.exit_orders.pop(sym, None)
        self.market_exits.pop(sym, None)

    def on_bar(self, sym: str, t: int) -> None:
        b = self.m.bar(sym, t)
        if b is None:
            self.entry_orders.pop(sym, None)
            return
        _, o, h, low, c = b
        tick = self.m.tick(sym, o)
        filled_now = False
        # market exits decided at the start of this bar
        if sym in self.market_exits and sym in self.pos:
            self.close_pos(sym, t, o - tick, TAKER, self.market_exits[sym])
        # maker entry
        order = self.entry_orders.pop(sym, None)
        if order and sym not in self.pos:
            if low < order["limit"]:
                px = order["limit"]
                self.pos[sym] = Pos(order["qty"], px, order["stop"], t, order["qty"] * px * MAKER)
                self.entries.append(t)
                filled_now = True
            else:
                self.res.bump("entry_unfilled")
        p = self.pos.get(sym)
        if p is None:
            self.last_px[sym] = c
            return
        # funding
        for ft, rate, mark in self.m.funding.get(sym, ()):
            if t <= ft < t + BAR_MS and not filled_now:
                p.funding += p.qty * (mark or o) * rate
        # stop
        if low <= p.stop:
            px = min(o, p.stop) if not filled_now else p.stop
            self.close_pos(sym, t, px - tick, TAKER, "stop")
            self.last_px[sym] = c
            return
        # maker exit
        ex = self.exit_orders.pop(sym, None)
        if ex is not None:
            if h > ex["limit"]:
                self.close_pos(sym, t, ex["limit"], MAKER, "model_close_maker")
            else:
                self.res.bump("exit_unfilled")
        self.last_px[sym] = c

    def day_roll(self, t: int) -> None:
        day = t // 86_400_000
        if day != self.day:
            self.day = day
            self.day_start = self.mtm()
            self.day_blocked = False
        if self.r.daily_loss and not self.day_blocked and self.day_start > 0:
            if (self.mtm() - self.day_start) / self.day_start <= -DAILY_LOSS:
                self.day_blocked = True
                self.res.bump("daily_loss_days")
                for sym in self.pos:
                    self.market_exits[sym] = "daily_loss"


def run_rules(rules: Rules, decisions: list[Decision], market: Market, equity: float, t0: int, t1: int) -> Result:
    sim = Sim(rules, market, equity)
    by_bar: dict[int, list[Decision]] = {}
    for d in decisions:
        by_bar.setdefault(d.bar_ms + BAR_MS, []).append(d)
    t = t0
    while t <= t1:
        sim.day_roll(t)
        for d in by_bar.get(t, ()):
            sim.on_decision(d, t)
        active = set(sim.pos) | set(sim.entry_orders) | set(sim.exit_orders) | set(sim.market_exits)
        for sym in sorted(active):
            sim.on_bar(sym, t)
        sim.res.curve.append(sim.mtm())
        t += BAR_MS
    # mark open positions to the last close as a taker exit
    for sym in list(sim.pos):
        px = sim.last_px.get(sym, sim.pos[sym].entry)
        sim.close_pos(sym, t, px - market.tick(sym, px), TAKER, "end_of_data")
    sim.res.curve.append(sim.mtm())
    return sim.res


def run_random(template: Result, decisions: list[Decision], market: Market, equity: float, seed: int, t1: int,
               maker_exit: bool = True, fee_budget: bool = True) -> Result:
    """Same trade count as template; entries on random decision slots; the template's sizing and exit style.

    Hold time is drawn from the template's own holds; ATR stop as in the sim. Equity for sizing is fixed at the
    start value and positions may overlap freely (no max_positions), so this is a rough null, not a replay.
    """
    rng = random.Random(seed)
    n = len(template.trades)
    holds = [max(BAR_MS, tr.t_out - tr.t_in) for tr in template.trades if tr.reason != "end_of_data"] or [BAR_MS * 6]
    res = Result(f"C#{seed}")
    tries = 0
    busy: dict[str, int] = {}
    while len(res.trades) < n and tries < n * 50:
        tries += 1
        d = decisions[rng.randrange(len(decisions))]
        sym = d.symbol
        t = d.bar_ms + BAR_MS
        if busy.get(sym, -1) >= t:
            continue
        s_bar = market.bar(sym, d.bar_ms)
        atr = market.atr.get(sym, {}).get(d.bar_ms)
        b = market.bar(sym, t)
        if not s_bar or not atr or not b:
            continue
        close = s_bar[4]
        if not (b[3] < close):
            continue  # maker entry would not fill
        stop = close - ATR_MULT * atr
        risk = equity * RISK_PCT
        qty = min(risk / (ATR_MULT * atr), equity * LEVERAGE / close)
        if fee_budget:
            qty = qty_after_fee_budget(qty=qty, price=close, risk_amount=risk)
        fee_in = qty * close * MAKER
        hold_until = t + rng.choice(holds)
        funding = 0.0
        tt = t
        exit_px, reason, fee_rate = None, "", TAKER
        while tt <= t1:
            bar = market.bar(sym, tt)
            if bar is None:
                tt += BAR_MS
                continue
            _, o, h, low, c = bar
            tick = market.tick(sym, o)
            for ft, rate, mark in market.funding.get(sym, ()):
                if tt <= ft < tt + BAR_MS and ft >= t:
                    funding += qty * (mark or o) * rate
            if low <= stop:
                exit_px, reason = (min(o, stop) if tt > t else stop) - tick, "stop"
                break
            if tt >= hold_until and not maker_exit:  # market exit at the open
                exit_px, reason = o - tick, "time_exit_market"
                break
            if tt >= hold_until:  # maker exit at the previous close
                prev = market.bar(sym, tt - BAR_MS)
                limit = prev[4] if prev else o
                if h > limit:
                    exit_px, reason, fee_rate = limit, "time_exit_maker", MAKER
                    break
            tt += BAR_MS
        if exit_px is None:
            last = market.bar(sym, tt - BAR_MS) or b
            exit_px, reason = last[4], "end_of_data"
        res.trades.append(Trade(sym, t, tt, qty, close, exit_px, qty * (exit_px - close), fee_in + qty * exit_px * fee_rate, funding, reason))
        busy[sym] = tt
    return res


def hold_btc(market: Market, equity: float, t0: int, t1: int) -> Result:
    res = Result("D")
    bars = market.bars.get("BTCUSDT", {})
    first = bars.get(t0)
    last_t = max(k for k in bars if k <= t1)
    last = bars[last_t]
    tick = market.tick("BTCUSDT", first[1])
    entry = first[1] + tick
    qty = equity / entry
    exit_px = last[4] - tick
    funding = sum(qty * (mark or entry) * rate for ft, rate, mark in market.funding.get("BTCUSDT", ()) if t0 <= ft <= last_t)
    res.trades.append(Trade("BTCUSDT", t0, last_t, qty, entry, exit_px, qty * (exit_px - entry), qty * entry * TAKER + qty * exit_px * TAKER, funding, "hold"))
    eq = equity
    for t in sorted(k for k in bars if t0 <= k <= last_t):
        res.curve.append(eq + qty * (bars[t][4] - entry))
    return res


# --- report -------------------------------------------------------------------

def summarize(res: Result, equity: float) -> dict[str, Any]:
    trades = res.trades
    nets = [t.net for t in trades]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    curve = res.curve or [equity]
    if not res.curve:
        acc = equity
        curve = [acc]
        for t in sorted(trades, key=lambda x: x.t_out):
            acc += t.net
            curve.append(acc)
    peak, dd, dd_pct = curve[0], 0.0, 0.0
    for v in curve:
        peak = max(peak, v)
        if peak - v > dd:
            dd = peak - v
            dd_pct = dd / peak if peak else 0.0
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    return {
        "name": res.name,
        "net": sum(nets),
        "price": sum(t.price_pnl for t in trades),
        "fees": sum(t.fees for t in trades),
        "funding": sum(t.funding for t in trades),
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "avg_win": statistics.mean(wins) if wins else 0.0,
        "avg_loss": statistics.mean(losses) if losses else 0.0,
        "max_dd": dd,
        "max_dd_pct": dd_pct,
        "exits": reasons,
        "counters": dict(res.counters),
    }


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def fmt_row(s: dict[str, Any]) -> str:
    return (
        f"| {s['name']} | {s['net']:+.2f} | {s['price']:+.2f} | {-s['fees']:.2f} | {-s['funding']:+.2f} | "
        f"{s['trades']} | {s['win_rate'] * 100:.1f}% | {s['avg_win']:+.2f} | {s['avg_loss']:+.2f} | "
        f"{s['max_dd']:.2f} ({s['max_dd_pct'] * 100:.1f}%) |"
    )


HEADER = (
    "| вариант | нетто | цена | комиссия | фандинг | сделок | плюсов | ср. плюс | ср. минус | max DD |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True, help="copy of the ledger sqlite (opened read-only)")
    ap.add_argument("--venue", choices=sorted(HOSTS), default="mainnet")
    ap.add_argument("--cache", default="/tmp/jev_bt_cache")
    ap.add_argument("--equity", type=float, default=None, help="start equity (default: first wallet value in the log)")
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--out", default=None, help="write the JSON summary here")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    decisions, logged_equity = load_decisions(args.ledger)
    if not decisions:
        raise SystemExit("no model answers in the ledger")
    equity = args.equity or logged_equity or 10000.0
    t0 = decisions[0].bar_ms + BAR_MS
    t1 = decisions[-1].bar_ms + BAR_MS
    symbols = sorted({d.symbol for d in decisions} | {"BTCUSDT"})
    market = load_market(args.venue, symbols, t0, t1, cache)
    have = {d.symbol for d in decisions if d.symbol in market.bars}
    used = [d for d in decisions if d.symbol in market.bars]

    variants = [
        old_rules("A"),
        old_rules("A+dl", daily_loss=True),
        Rules("B@0.60", threshold=0.60),
        Rules("B@0.72", threshold=0.72),
        Rules("B@0.80", threshold=0.80),
        Rules("B@0.72 jev", threshold=0.72, models="jev"),
        Rules("B@0.72 laya", threshold=0.72, models="laya"),
        # Informational: the logged buy answers never reach 0.55, so 0.60+ trade nothing.
        Rules("B@0.45", threshold=0.45),
        Rules("B@0.40", threshold=0.40),
        # B mechanics (maker exit, fee sizing, caps, window, pause, daily_loss) on A's entries.
        Rules("B-mech", follow_jev=True),
    ]
    results = {v.name: run_rules(v, used, market, equity, t0, t1) for v in variants}
    summaries = [summarize(r, equity) for r in results.values()]

    rand_blocks = {}
    for base in ("B@0.72", "B@0.40", "B-mech", "A"):
        tmpl = results[base]
        if not tmpl.trades:
            continue
        rv = next(v for v in variants if v.name == base)
        sums = [
            summarize(run_random(tmpl, used, market, equity, seed, t1, maker_exit=rv.maker_exit, fee_budget=rv.fee_budget), equity)
            for seed in range(args.seeds)
        ]
        nets = [s["net"] for s in sums]
        base_net = summarize(tmpl, equity)["net"]
        rand_blocks[base] = {
            "trades": len(tmpl.trades),
            "seeds": args.seeds,
            "net_p5": pct(nets, 0.05),
            "net_median": pct(nets, 0.5),
            "net_p95": pct(nets, 0.95),
            "win_rate_median": pct([s["win_rate"] for s in sums], 0.5),
            "fees_median": pct([s["fees"] for s in sums], 0.5),
            "funding_median": pct([s["funding"] for s in sums], 0.5),
            "price_median": pct([s["price"] for s in sums], 0.5),
            "max_dd_median": pct([s["max_dd"] for s in sums], 0.5),
            "share_random_at_least_base": sum(1 for x in nets if x >= base_net) / len(nets),
        }
    d_sum = summarize(hold_btc(market, equity, t0, t1), equity)

    iso = lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")  # noqa: E731
    models: dict[str, int] = {}
    for d in used:
        models[d.model] = models.get(d.model, 0) + 1
    meta = {
        "venue": args.venue,
        "period": [iso(t0), iso(t1)],
        "start_equity": equity,
        "answers_total": len(decisions),
        "answers_used": len(used),
        "answers_by_model": models,
        "symbols_used": sorted(have),
        "symbols_without_candles": sorted({d.symbol for d in decisions} - have),
        "funding_missing": market.funding_missing,
        "fees": {"taker": TAKER, "maker": MAKER, "market_slippage_ticks": 1},
    }
    lines = [f"# Replay {args.venue}: {meta['period'][0]} .. {meta['period'][1]}, equity {equity:.2f}", "", HEADER]
    lines += [fmt_row(s) for s in summaries]
    lines.append(fmt_row(d_sum | {"name": "D hold BTC"}))
    for base, blk in rand_blocks.items():
        lines.append(
            f"\nC (random, as many trades as {base}: {blk['trades']}, {blk['seeds']} seeds): net p5 {blk['net_p5']:+.2f}, "
            f"median {blk['net_median']:+.2f}, p95 {blk['net_p95']:+.2f}; median price {blk['price_median']:+.2f}, "
            f"fees {-blk['fees_median']:.2f}, funding {-blk['funding_median']:+.2f}, win {blk['win_rate_median'] * 100:.1f}%, "
            f"max DD {blk['max_dd_median']:.2f}; random >= {base} in {blk['share_random_at_least_base'] * 100:.0f}% of seeds"
        )
    lines.append("\nexits / guards:")
    for s in summaries:
        lines.append(f"- {s['name']}: exits {s['exits']}; counters {s['counters']}")
    buys = [d.judgment.should_trade_now for d in used if d.judgment.action == "buy_long"]
    meta["buy_answers"] = len(buys)
    meta["buy_should_trade_now_max"] = max(buys) if buys else None
    lines.append(f"\nmeta: {json.dumps(meta, ensure_ascii=False)}")
    print("\n".join(lines))
    if args.out:
        Path(args.out).write_text(json.dumps({"meta": meta, "variants": summaries, "random": rand_blocks, "hold_btc": d_sum}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
