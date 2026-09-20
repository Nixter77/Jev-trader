from __future__ import annotations

import json
from pathlib import Path

from jev_trader.public_market import UniverseTicker, parse_ticker_24hr
from jev_trader.universe import is_usdt_perp, select_trade_universe

FIXTURES = Path(__file__).parent / "fixtures"


def test_dated_and_non_usdt_are_not_perps() -> None:
    assert is_usdt_perp("BTCUSDT")
    assert is_usdt_perp("ETHUSDT")
    assert not is_usdt_perp("BTCUSDT_261225")
    assert not is_usdt_perp("ETHUSDC")
    assert not is_usdt_perp("BTCUSD")


def test_select_trade_universe_ranks_volume_keeps_opens() -> None:
    raw = json.loads((FIXTURES / "ticker_24hr_sample.json").read_text(encoding="utf-8"))
    rows = parse_ticker_24hr(raw)
    rows.append(
        UniverseTicker(
            symbol="BTCUSDT_261225",
            last=1.0,
            price_change_percent=0.0,
            quote_volume=9e12,
        )
    )
    chosen = select_trade_universe(rows, limit=3, min_quote_volume=0)
    assert "BTCUSDT" in chosen
    assert "BTCUSDT_261225" not in chosen
    assert all("_" not in symbol for symbol in chosen)
    with_open = select_trade_universe(
        rows, limit=2, min_quote_volume=0, extra=["RAREUSDT"]
    )
    assert with_open[0] == "RAREUSDT"
    assert "BTCUSDT_261225" not in with_open
