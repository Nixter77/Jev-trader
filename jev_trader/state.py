from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jev_trader.features import compute_features
from jev_trader.models import CompactState, Features, MarketSnapshot


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _compact_payload(snapshot: MarketSnapshot, features: Features) -> dict[str, Any]:
    pos = snapshot.position
    return {
        "symbol": snapshot.symbol,
        "tf": snapshot.tf,
        "ts": _iso(features.ts),
        "price": {
            "close": round(features.close, 4),
            "ret_1": None if features.ret_1 is None else round(features.ret_1, 4),
            "ret_5": None if features.ret_5 is None else round(features.ret_5, 4),
            "ret_12": None if features.ret_12 is None else round(features.ret_12, 4),
        },
        "vol": {
            "atr14_pct": None if features.atr_pct is None else round(features.atr_pct, 4),
            "bar_range_atr": None
            if features.bar_range_atr is None
            else round(features.bar_range_atr, 4),
        },
        "trend": {
            "ema20": None if features.ema20 is None else round(features.ema20, 4),
            "ema50": None if features.ema50 is None else round(features.ema50, 4),
            "ema200": None if features.ema200 is None else round(features.ema200, 4),
            "ema_stack": features.ema_stack,
            "adx": None if features.adx is None else round(features.adx, 2),
        },
        "osc": {
            "rsi": None if features.rsi is None else round(features.rsi, 2),
            "vwap_dist_pct": None
            if features.vwap_dist_pct is None
            else round(features.vwap_dist_pct, 4),
        },
        "structure": {
            "pattern": features.structure,
            "last_swing_low_pct": None
            if features.last_swing_low_pct is None
            else round(features.last_swing_low_pct, 4),
            "last_swing_high_pct": None
            if features.last_swing_high_pct is None
            else round(features.last_swing_high_pct, 4),
        },
        "book": {
            "imbalance": None
            if features.book_imbalance is None
            else round(features.book_imbalance, 4),
            "spread_bps": None if features.spread_bps is None else round(features.spread_bps, 4),
        },
        "position": {
            "side": pos.side,
            "size": pos.size,
            "entry": pos.entry,
            "upnl_pct": pos.upnl_pct,
            "bars_in_trade": pos.bars_in_trade,
            "cash_usdt": pos.cash_usdt,
        },
        "macro": {
            "funding": snapshot.funding,
            "doi_1h": snapshot.doi_1h,
            "btc_corr": snapshot.btc_corr,
        },
        "news": list(snapshot.news),
    }


def _compact_text(snapshot: MarketSnapshot, features: Features) -> str:
    pos = snapshot.position
    news = "; ".join(snapshot.news) if snapshot.news else "none"
    vwap_s = features.vwap_dist_pct
    vwap_fmt = "n/a" if vwap_s is None else f"{vwap_s:+.2f}%"
    imb = features.book_imbalance
    imb_fmt = "n/a" if imb is None else f"{imb:+.2f}"
    return (
        f"symbol={snapshot.symbol} tf={snapshot.tf} ts={_iso(features.ts)}\n"
        f"close={_fmt(features.close, 2)} ret_1={_fmt(features.ret_1)}% "
        f"ret_5={_fmt(features.ret_5)}% ret_12={_fmt(features.ret_12)}% "
        f"atr14={_fmt(features.atr_pct)}%\n"
        f"{features.ema_stack} adx={_fmt(features.adx, 0)} rsi={_fmt(features.rsi, 0)} "
        f"vwap_dist={vwap_fmt}\n"
        f"structure={features.structure} last_swing_low={_fmt(features.last_swing_low_pct)}%\n"
        f"book_imb={imb_fmt} spread={_fmt(features.spread_bps, 1)}bps\n"
        f"pos={pos.side} size={pos.size} cash_usdt={pos.cash_usdt}\n"
        f"news: {news}"
    )


def build_compact_state(
    snapshot: MarketSnapshot, features: Features | None = None
) -> CompactState:
    """Serialize OHLCV + optional book/position/news into a small JSON/text state.

    Raw candles stay out of the payload. Jev is text-only: no chart images.
    """
    feats = features if features is not None else compute_features(snapshot)
    payload = _compact_payload(snapshot, feats)
    if "candles" in payload:
        raise RuntimeError("compact state must not include raw candles")
    return CompactState(payload=payload, text=_compact_text(snapshot, feats))
