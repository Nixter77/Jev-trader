from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from jev_trader.execution import BinanceTestnetBroker
from jev_trader.public_market import (
    KLINES_PATH,
    MINI_TICKER_STREAM,
    PUBLIC_REST_HOST,
    SIGNED_ORDER_PATH,
    TICKER_24HR_PATH,
    assert_unsigned_request,
    build_klines_request,
    build_public_get_request,
    build_ticker_24hr_request,
    compact_universe_payload,
    fetch_klines,
    fetch_universe_summary,
    parse_rest_klines,
    parse_ticker_24hr,
    parse_ws_kline_event,
    public_kline_stream,
    public_ws_url,
)

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(payload).encode() if not isinstance(payload, (bytes, bytearray)) else payload

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, payload: Any) -> list[Any]:
    captured: list[Any] = []

    def fake_urlopen(req: Any, timeout: float | None = None, context: Any = None) -> _FakeResponse:
        captured.append({"req": req, "timeout": timeout, "context": context})
        return _FakeResponse(payload)

    monkeypatch.setattr("jev_trader.public_market.urllib.request.urlopen", fake_urlopen)
    return captured


def _assert_no_credentials(req: Any, api_key: str, api_secret: str) -> None:
    parsed = urllib.parse.urlparse(req.full_url)
    qs = urllib.parse.parse_qs(parsed.query)
    header_map = {str(k).lower(): str(v) for k, v in req.header_items()}
    blob = req.full_url + "".join(header_map.values())
    assert parsed.hostname == PUBLIC_REST_HOST
    assert parsed.path != SIGNED_ORDER_PATH
    assert "signature" not in qs
    assert "x-mbx-apikey" not in header_map
    assert api_key not in blob
    assert api_secret not in blob
    assert req.get_method() == "GET"


def test_ticker_24hr_request_is_public_unsigned_path() -> None:
    req = build_ticker_24hr_request()
    parsed = urllib.parse.urlparse(req.full_url)
    assert parsed.scheme == "https"
    assert parsed.hostname == "fapi.binance.com"
    assert parsed.path == TICKER_24HR_PATH
    assert parsed.path != SIGNED_ORDER_PATH
    assert req.get_method() == "GET"
    assert "signature" not in (parsed.query or "")
    header_map = {str(k).lower(): str(v) for k, v in req.header_items()}
    assert "x-mbx-apikey" not in header_map
    assert_unsigned_request(req)


def test_build_public_get_strips_hmac_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_API_KEY", "leak-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "leak-secret")
    req = build_public_get_request(
        TICKER_24HR_PATH,
        {"signature": "deadbeef", "timestamp": "1", "recvWindow": "5000"},
    )
    assert "signature" not in req.full_url
    assert "timestamp" not in req.full_url.lower()
    assert "deadbeef" not in req.full_url
    _assert_no_credentials(req, "leak-key", "leak-secret")


def test_fetch_universe_summary_does_not_send_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "binance-key-must-not-leak"
    api_secret = "binance-secret-must-not-leak"
    monkeypatch.setenv("BINANCE_API_KEY", api_key)
    monkeypatch.setenv("BINANCE_API_SECRET", api_secret)
    sample = json.loads((FIXTURES / "ticker_24hr_sample.json").read_text(encoding="utf-8"))
    captured = _patch_urlopen(monkeypatch, sample)
    rows = fetch_universe_summary()
    assert len(captured) == 1
    req = captured[0]["req"]
    parsed = urllib.parse.urlparse(req.full_url)
    assert parsed.path == TICKER_24HR_PATH
    assert parsed.path != SIGNED_ORDER_PATH
    _assert_no_credentials(req, api_key, api_secret)
    assert any(row.symbol == "BTCUSDT" for row in rows)
    assert all(isinstance(row.last, float) for row in rows)


def test_fetch_klines_request_is_unsigned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    payload = json.loads((FIXTURES / "klines_5m.json").read_text(encoding="utf-8"))
    captured = _patch_urlopen(monkeypatch, payload)
    rows = fetch_klines("BTCUSDT", interval="5m", limit=240)
    req = captured[0]["req"]
    parsed = urllib.parse.urlparse(req.full_url)
    assert parsed.path == KLINES_PATH
    qs = urllib.parse.parse_qs(parsed.query)
    assert qs["symbol"] == ["BTCUSDT"]
    assert qs["interval"] == ["5m"]
    assert "signature" not in qs
    assert len(rows) == len(payload)


def test_signed_trading_still_refuses_production() -> None:
    with pytest.raises(RuntimeError, match="production"):
        BinanceTestnetBroker("k", "s", "https://fapi.binance.com")
    with pytest.raises(RuntimeError, match="production"):
        BinanceTestnetBroker("k", "s", "https://fstream.binance.com")


def test_ws_subscribe_strings_are_public_and_unsigned() -> None:
    mini = public_ws_url(MINI_TICKER_STREAM)
    kline = public_ws_url(public_kline_stream("BTCUSDT", "5m"))
    assert mini == "wss://fstream.binance.com/ws/!miniTicker@arr"
    assert kline == "wss://fstream.binance.com/ws/btcusdt@kline_5m"
    assert "signature" not in mini
    assert "X-MBX-APIKEY" not in kline
    req = build_klines_request("ETHUSDT")
    assert urllib.parse.urlparse(req.full_url).path == KLINES_PATH


def test_parse_recorded_ticker_24hr_and_mini() -> None:
    rest = json.loads((FIXTURES / "ticker_24hr_sample.json").read_text(encoding="utf-8"))
    rows = parse_ticker_24hr(rest)
    by_symbol = {row.symbol: row for row in rows}
    assert "BTCUSDT" in by_symbol
    assert "ETHUSDT" in by_symbol
    assert by_symbol["BTCUSDT"].last > 0
    assert by_symbol["BTCUSDT"].price_change_percent is not None
    mini = json.loads((FIXTURES / "mini_ticker_arr.json").read_text(encoding="utf-8"))
    mini_rows = parse_ticker_24hr(mini)
    assert {row.symbol for row in mini_rows} >= {"BTCUSDT", "ETHUSDT"}
    assert all(isinstance(row.last, float) for row in mini_rows)


def test_compact_universe_payload_keeps_count_and_watch() -> None:
    rest = json.loads((FIXTURES / "ticker_24hr_sample.json").read_text(encoding="utf-8"))
    rows = parse_ticker_24hr(rest)
    compact = compact_universe_payload(rows)
    assert compact["ok"] is True
    assert compact["authenticated"] is False
    assert compact["count"] == len(rows)
    assert compact["count"] > 1
    assert "tickers" not in compact
    assert compact["watch"]["BTCUSDT"]["last"] > 0
    assert isinstance(compact["watch"]["BTCUSDT"]["price_change_percent"], float)
    assert compact["gainers"]
    assert compact["losers"]


def test_parse_recorded_klines_and_ws_events() -> None:
    raw = json.loads((FIXTURES / "klines_5m.json").read_text(encoding="utf-8"))
    rows = parse_rest_klines(raw)
    assert len(rows) == 240
    assert rows[0].candle.ts == int(raw[0][0])
    assert rows[0].candle.close == float(raw[0][4])
    opened = parse_ws_kline_event(
        json.loads((FIXTURES / "kline_ws_open.json").read_text(encoding="utf-8"))
    )
    closed = parse_ws_kline_event(
        json.loads((FIXTURES / "kline_ws_closed.json").read_text(encoding="utf-8"))
    )
    closed2 = parse_ws_kline_event(
        json.loads((FIXTURES / "kline_ws_closed_2.json").read_text(encoding="utf-8"))
    )
    assert opened is not None and opened.closed is False
    assert closed is not None and closed.closed is True
    assert closed2 is not None and closed2.closed is True
    assert closed.candle.ts != closed2.candle.ts
    assert opened.candle.ts == closed.candle.ts
