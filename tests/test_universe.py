from __future__ import annotations

import json
from pathlib import Path

from jev_trader.public_market import UniverseTicker, parse_ticker_24hr
from jev_trader.universe import (
    DEFAULT_LAYA_UNIVERSE_SIZE,
    DEFAULT_UNIVERSE_SIZE,
    is_usdt_perp,
    order_symbols_open_first,
    resolve_universe_size,
    select_trade_universe,
)

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


def test_order_symbols_open_first_stable() -> None:
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
    assert order_symbols_open_first(symbols, ["CCCUSDT", "ZZZUSDT", "AAAUSDT"]) == [
        "CCCUSDT",
        "ZZZUSDT",
        "AAAUSDT",
        "BBBUSDT",
        "DDDUSDT",
    ]
    assert order_symbols_open_first(symbols, []) == symbols
    assert order_symbols_open_first(symbols, ["bbbusdt", "BBBUsDT"]) == [
        "BBBUSDT",
        "AAAUSDT",
        "CCCUSDT",
        "DDDUSDT",
    ]


def test_resolve_universe_size_laya_vs_jev() -> None:
    assert resolve_universe_size(backend="jev") == DEFAULT_UNIVERSE_SIZE
    assert resolve_universe_size(backend="laya", environ={}) == DEFAULT_LAYA_UNIVERSE_SIZE
    assert resolve_universe_size(backend="laya", environ={"LAYA_UNIVERSE_SIZE": "7"}) == 7
    assert resolve_universe_size(backend="laya", explicit=12, environ={"LAYA_UNIVERSE_SIZE": "3"}) == 12
    assert resolve_universe_size(backend="jev", explicit=None) == DEFAULT_UNIVERSE_SIZE
