from __future__ import annotations

from jev_trader.models import BookLevel, Candle, Features, MarketSnapshot, OrderBook


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None
    alpha = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for price in values[period:]:
        ema = alpha * price + (1.0 - alpha) * ema
    return ema


def _true_ranges(candles: list[Candle]) -> list[float]:
    trs: list[float] = []
    prev_close = candles[0].close
    for c in candles:
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
        prev_close = c.close
    return trs


def _wilder_smooth(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    acc = sum(values[:period])
    out[period - 1] = acc / period
    for i in range(period, len(values)):
        acc = acc - (acc / period) + values[i]
        out[i] = acc / period
    return out


def _atr(candles: list[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    smoothed = _wilder_smooth(_true_ranges(candles), period)
    return smoothed[-1]


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(candles: list[Candle], period: int = 14) -> float | None:
    if len(candles) < period * 2:
        return None
    plus_dm: list[float] = [0.0]
    minus_dm: list[float] = [0.0]
    for i in range(1, len(candles)):
        up = candles[i].high - candles[i - 1].high
        down = candles[i - 1].low - candles[i].low
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    tr_s = _wilder_smooth(_true_ranges(candles), period)
    plus_s = _wilder_smooth(plus_dm, period)
    minus_s = _wilder_smooth(minus_dm, period)
    dx: list[float] = []
    dx_index: list[int] = []
    for i, (tr, p, m) in enumerate(zip(tr_s, plus_s, minus_s)):
        if tr is None or p is None or m is None or tr == 0:
            continue
        plus_di = 100.0 * p / tr
        minus_di = 100.0 * m / tr
        denom = plus_di + minus_di
        if denom == 0:
            continue
        dx.append(100.0 * abs(plus_di - minus_di) / denom)
        dx_index.append(i)
    if len(dx) < period:
        return None
    adx_series = _wilder_smooth(dx, period)
    return adx_series[-1]


def _vwap(candles: list[Candle]) -> float | None:
    num = 0.0
    den = 0.0
    for c in candles:
        typical = (c.high + c.low + c.close) / 3.0
        num += typical * c.volume
        den += c.volume
    if den == 0:
        return None
    return num / den


def _pct_return(closes: list[float], bars: int) -> float | None:
    if len(closes) <= bars:
        return None
    prev = closes[-1 - bars]
    if prev == 0:
        return None
    return (closes[-1] - prev) / prev * 100.0


def _swings(candles: list[Candle], left: int = 3, right: int = 3) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    n = len(candles)
    for i in range(left, n - right):
        window = candles[i - left : i + right + 1]
        hi = candles[i].high
        lo = candles[i].low
        if all(hi >= c.high for c in window):
            highs.append((i, hi))
        if all(lo <= c.low for c in window):
            lows.append((i, lo))
    return highs, lows


def _structure(candles: list[Candle], close: float) -> tuple[str, float | None, float | None]:
    highs, lows = _swings(candles)
    if len(highs) < 2 or len(lows) < 2:
        last_low = (lows[-1][1] - close) / close * 100.0 if lows and close else None
        last_high = (highs[-1][1] - close) / close * 100.0 if highs and close else None
        return "unknown", last_low, last_high
    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    lh = highs[-1][1] < highs[-2][1]
    ll = lows[-1][1] < lows[-2][1]
    if hh and hl:
        pattern = "HH_HL"
    elif lh and ll:
        pattern = "LH_LL"
    elif hh and ll:
        pattern = "HH_LL"
    elif lh and hl:
        pattern = "LH_HL"
    else:
        pattern = "mixed"
    last_swing_low_pct = (lows[-1][1] - close) / close * 100.0 if close else None
    last_swing_high_pct = (highs[-1][1] - close) / close * 100.0 if close else None
    return pattern, last_swing_low_pct, last_swing_high_pct


def _ema_stack(ema20: float | None, ema50: float | None, ema200: float | None) -> str:
    labeled = [("ema20", ema20), ("ema50", ema50), ("ema200", ema200)]
    present = [(name, value) for name, value in labeled if value is not None]
    if len(present) < 2:
        return "n/a"
    ordered = sorted(present, key=lambda item: item[1], reverse=True)
    return ">".join(name for name, _ in ordered)


def book_imbalance_and_spread(book: OrderBook | None, top: int = 5) -> tuple[float | None, float | None]:
    if book is None or not book.bids or not book.asks:
        return None, None
    bids = book.bids[:top]
    asks = book.asks[:top]
    bid_qty = sum(level.qty for level in bids)
    ask_qty = sum(level.qty for level in asks)
    denom = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / denom if denom else None
    best_bid = book.bids[0].price
    best_ask = book.asks[0].price
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10_000.0 if mid else None
    return imbalance, spread_bps


def compute_features(snapshot: MarketSnapshot) -> Features:
    candles = list(snapshot.candles)
    if not candles:
        raise ValueError("snapshot has no candles")
    closes = [c.close for c in candles]
    close = closes[-1]
    atr = _atr(candles)
    atr_pct = (atr / close * 100.0) if atr and close else None
    last = candles[-1]
    bar_range = last.high - last.low
    bar_range_atr = (bar_range / atr) if atr else None
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    ema200 = _ema(closes, 200)
    vwap = _vwap(candles)
    vwap_dist = ((close - vwap) / vwap * 100.0) if vwap else None
    structure, last_low_pct, last_high_pct = _structure(candles, close)
    imbalance, spread = book_imbalance_and_spread(snapshot.book)
    return Features(
        close=close,
        ret_1=_pct_return(closes, 1),
        ret_5=_pct_return(closes, 5),
        ret_12=_pct_return(closes, 12),
        atr=atr,
        atr_pct=atr_pct,
        bar_range_atr=bar_range_atr,
        ema20=ema20,
        ema50=ema50,
        ema200=ema200,
        ema_stack=_ema_stack(ema20, ema50, ema200),
        adx=_adx(candles),
        rsi=_rsi(closes),
        vwap=vwap,
        vwap_dist_pct=vwap_dist,
        structure=structure,
        last_swing_low_pct=last_low_pct,
        last_swing_high_pct=last_high_pct,
        book_imbalance=imbalance,
        spread_bps=spread,
        ts=last.ts,
    )


def parse_book(raw: dict | None) -> OrderBook | None:
    if not raw:
        return None
    bids = tuple(BookLevel(price=float(p), qty=float(q)) for p, q in raw.get("bids", []))
    asks = tuple(BookLevel(price=float(p), qty=float(q)) for p, q in raw.get("asks", []))
    if not bids and not asks:
        return None
    return OrderBook(bids=bids, asks=asks)
