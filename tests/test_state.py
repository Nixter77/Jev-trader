from __future__ import annotations

import json

from jev_trader.features import compute_features
from jev_trader.snapshot import build_fixture_snapshot
from jev_trader.state import build_compact_state


REQUIRED_BLOCKS = ("price", "vol", "trend", "osc", "structure", "book", "position")


def test_compact_state_has_named_blocks_and_is_small(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    compact = build_compact_state(market_snapshot, features)
    payload = compact.as_dict()

    for block in REQUIRED_BLOCKS:
        assert block in payload, f"missing state block {block}"

    price = payload["price"]
    assert "close" in price
    assert "ret_1" in price and "ret_5" in price and "ret_12" in price

    assert payload["vol"]["atr14_pct"] is not None
    trend = payload["trend"]
    assert trend["ema20"] is not None
    assert trend["ema50"] is not None
    assert trend["ema200"] is not None
    assert "ema" in trend["ema_stack"]
    assert trend["adx"] is not None

    osc = payload["osc"]
    assert osc["rsi"] is not None
    assert osc["vwap_dist_pct"] is not None

    structure = payload["structure"]
    assert structure["pattern"] in {"HH_HL", "LH_LL", "HH_LL", "LH_HL", "mixed", "unknown"}
    assert "last_swing_low_pct" in structure

    book = payload["book"]
    assert book["imbalance"] is not None
    assert book["spread_bps"] is not None

    position = payload["position"]
    assert position["side"] == "FLAT"
    assert position["cash_usdt"] == 10000

    assert "candles" not in payload
    dumped = json.dumps(payload)
    raw_500 = json.dumps(
        [
            {
                "ts": c.ts,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in market_snapshot.candles
        ]
        + [
            {
                "ts": i,
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
            }
            for i in range(500)
        ]
    )
    assert len(dumped) * 8 < len(raw_500)
    assert len(compact.as_text()) < 2_000
    assert "symbol=BTCUSDT" in compact.as_text()


def test_uptrend_fixture_has_hh_hl_or_stack(market_snapshot) -> None:
    compact = build_compact_state(market_snapshot).as_dict()
    assert compact["trend"]["ema_stack"].startswith("ema20")
    assert compact["structure"]["pattern"] in {"HH_HL", "mixed", "HH_LL"}


def test_book_and_position_fixture_drive_shipped_transform() -> None:
    snapshot = build_fixture_snapshot(side="LONG", size=0.01)
    compact = build_compact_state(snapshot).as_dict()
    assert compact["position"]["side"] == "LONG"
    assert compact["position"]["size"] == 0.01
    assert compact["book"]["imbalance"] > 0
