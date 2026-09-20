from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from jev_trader.features import parse_book
from jev_trader.models import Candle, MarketSnapshot, Position, PositionSide


def snapshot_from_dict(data: dict[str, Any]) -> MarketSnapshot:
    candles = tuple(
        Candle(
            ts=int(item["ts"]),
            open=float(item["open"]),
            high=float(item["high"]),
            low=float(item["low"]),
            close=float(item["close"]),
            volume=float(item["volume"]),
        )
        for item in data["candles"]
    )
    pos_raw = data.get("position") or {}
    side = str(pos_raw.get("side") or "FLAT")
    if side not in {"FLAT", "LONG", "SHORT"}:
        side = "FLAT"
    position = Position(
        side=side,  # type: ignore[arg-type]
        size=float(pos_raw.get("size") or 0.0),
        cash_usdt=float(pos_raw.get("cash_usdt") or 10_000.0),
        entry=pos_raw.get("entry"),
        upnl_pct=pos_raw.get("upnl_pct"),
        bars_in_trade=int(pos_raw.get("bars_in_trade") or 0),
        stop_price=pos_raw.get("stop_price"),
    )
    news = data.get("news") or []
    if isinstance(news, str):
        news_tuple = (news,)
    else:
        news_tuple = tuple(str(item) for item in news)
    return MarketSnapshot(
        symbol=str(data.get("symbol") or "BTCUSDT"),
        tf=str(data.get("tf") or "5m"),
        candles=candles,
        position=position,
        book=parse_book(data.get("book")),
        news=news_tuple,
        funding=data.get("funding"),
        doi_1h=data.get("doi_1h"),
        btc_corr=data.get("btc_corr"),
    )


def load_snapshot(path: str | Path) -> MarketSnapshot:
    with Path(path).open(encoding="utf-8") as fh:
        return snapshot_from_dict(json.load(fh))


def make_uptrend_candles(
    n: int = 240,
    start: float = 100_000.0,
    seed: int = 42,
    start_ts: int = 1_726_665_000_000,
    step_ms: int = 300_000,
) -> list[Candle]:
    rng = random.Random(seed)
    price = start
    candles: list[Candle] = []
    for i in range(n):
        drift = 12.0
        noise = rng.uniform(-35.0, 50.0)
        open_ = price
        close = price + drift + noise
        high = max(open_, close) + rng.uniform(8.0, 28.0)
        low = min(open_, close) - rng.uniform(8.0, 28.0)
        volume = rng.uniform(80.0, 220.0)
        candles.append(
            Candle(
                ts=start_ts + i * step_ms,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
        price = close
    return candles


def default_book(close: float) -> dict[str, Any]:
    tick = 0.5
    bids = [[close - tick * (i + 1), 2.0 + i * 0.4] for i in range(5)]
    asks = [[close + tick * (i + 1), 1.4 + i * 0.25] for i in range(5)]
    return {"bids": bids, "asks": asks}


def build_fixture_snapshot(
    n: int = 240,
    side: PositionSide = "FLAT",
    size: float = 0.0,
) -> MarketSnapshot:
    candles = make_uptrend_candles(n=n)
    close = candles[-1].close
    return snapshot_from_dict(
        {
            "symbol": "BTCUSDT",
            "tf": "5m",
            "candles": [c.__dict__ for c in candles],
            "book": default_book(close),
            "position": {
                "side": side,
                "size": size,
                "entry": close * 0.99 if side != "FLAT" else None,
                "upnl_pct": 0.4 if side != "FLAT" else None,
                "bars_in_trade": 3 if side != "FLAT" else 0,
                "cash_usdt": 10_000,
            },
            "news": ["ETF inflow +$240m; no FOMC today"],
            "funding": 0.0001,
            "doi_1h": 0.02,
            "btc_corr": 1.0,
        }
    )


def snapshot_to_jsonable(snapshot: MarketSnapshot) -> dict[str, Any]:
    book = None
    if snapshot.book is not None:
        book = {
            "bids": [[lvl.price, lvl.qty] for lvl in snapshot.book.bids],
            "asks": [[lvl.price, lvl.qty] for lvl in snapshot.book.asks],
        }
    return {
        "symbol": snapshot.symbol,
        "tf": snapshot.tf,
        "candles": [
            {
                "ts": c.ts,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in snapshot.candles
        ],
        "book": book,
        "position": {
            "side": snapshot.position.side,
            "size": snapshot.position.size,
            "entry": snapshot.position.entry,
            "upnl_pct": snapshot.position.upnl_pct,
            "bars_in_trade": snapshot.position.bars_in_trade,
            "cash_usdt": snapshot.position.cash_usdt,
            "stop_price": snapshot.position.stop_price,
        },
        "news": list(snapshot.news),
        "funding": snapshot.funding,
        "doi_1h": snapshot.doi_1h,
        "btc_corr": snapshot.btc_corr,
    }
