from __future__ import annotations

import json
import urllib.parse
from typing import Any

import pytest

from jev_trader.execution import (
    BinanceTestnetBroker,
    PaperBroker,
    format_binance_decimal,
    is_real_fill,
    parse_symbol_filters,
    parse_usdt_wallet,
    round_to_step,
)
from jev_trader.models import ExecutionResult, TradeIntent


def _intent(**overrides) -> TradeIntent:
    base = dict(
        action="buy_long",
        qty=0.01,
        stop_price=99_000.0,
        stop_distance=100.0,
        entry_type="LIMIT_POST_ONLY",
        reduce_only=False,
        client_order_id="jev1_test",
        symbol="BTCUSDT",
        risk_pct=0.005,
        order_side="BUY",
        limit_price=104_900.5,
    )
    base.update(overrides)
    return TradeIntent(**base)


def test_paper_broker_records_without_network() -> None:
    result = PaperBroker().submit(_intent())
    assert result.venue == "paper"
    assert result.status == "paper_recorded"
    assert result.reduce_only is False
    assert result.detail["limit_price"] == 104_900.5


def test_testnet_broker_refuses_production() -> None:
    with pytest.raises(RuntimeError, match="production"):
        BinanceTestnetBroker("k", "s", "https://fapi.binance.com")


def test_live_broker_refuses_production_without_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    from jev_trader.execution import BinanceFuturesBroker

    monkeypatch.delenv("BINANCE_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="BINANCE_ALLOW_LIVE"):
        BinanceFuturesBroker("k", "s", "https://fapi.binance.com", live=True)


def test_live_broker_accepts_production_with_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    from jev_trader.execution import BinanceFuturesBroker

    monkeypatch.setenv("BINANCE_ALLOW_LIVE", "I_UNDERSTAND")
    broker = BinanceFuturesBroker("k", "s", "https://fapi.binance.com", live=True)
    assert broker.venue == "binance_live"
    assert "fapi.binance.com" in broker.base_url


class _FakeResponse:
    def __init__(self, payload: Any = None, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps({"orderId": 1} if payload is None else payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    captured: list[Any] = []

    def fake_urlopen(req: Any, timeout: float | None = None, context: Any = None) -> _FakeResponse:
        captured.append({"req": req, "timeout": timeout, "context": context})
        parsed = urllib.parse.urlparse(req.full_url)
        path = parsed.path
        method = req.get_method()
        params = urllib.parse.parse_qs(parsed.query)
        if path.endswith("/time"):
            return _FakeResponse({"serverTime": 1_000_000_000_000})
        if path.endswith("/exchangeInfo"):
            return _FakeResponse({"symbols": []})
        if path.endswith("/openOrders"):
            return _FakeResponse([])
        if method == "DELETE":
            return _FakeResponse({"code": 200})
        if method == "POST" and path.endswith("/order"):
            qty = (params.get("quantity") or ["0"])[0]
            if params.get("type") == ["MARKET"] or params.get("type") == ["STOP_MARKET"]:
                return _FakeResponse(
                    {
                        "orderId": 1,
                        "status": "FILLED",
                        "executedQty": qty,
                        "avgPrice": "100",
                        "symbol": (params.get("symbol") or ["BTCUSDT"])[0],
                    }
                )
            return _FakeResponse(
                {
                    "orderId": 1,
                    "status": "NEW",
                    "executedQty": "0",
                    "avgPrice": "0",
                    "symbol": (params.get("symbol") or ["BTCUSDT"])[0],
                    "timeInForce": (params.get("timeInForce") or [None])[0],
                }
            )
        return _FakeResponse()

    monkeypatch.setattr("jev_trader.execution.urllib.request.urlopen", fake_urlopen)
    return captured


def test_testnet_limit_post_only_sends_mandatory_price(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_urlopen(monkeypatch)
    broker = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    intent = _intent(limit_price=104_900.5, qty=0.0123, entry_type="LIMIT_POST_ONLY")
    result = broker.submit(intent)

    assert result.status == "working"
    assert not is_real_fill(result)
    posts = [c["req"] for c in captured if c["req"].get_method() == "POST"]
    assert len(posts) == 1
    req = posts[0]
    assert req.get_method() == "POST"
    parsed = urllib.parse.urlparse(req.full_url)
    assert parsed.netloc == "testnet.binancefuture.com"
    assert parsed.path == "/fapi/v1/order"
    params = urllib.parse.parse_qs(parsed.query)
    assert params["type"] == ["LIMIT"]
    assert params["timeInForce"] == ["GTX"]
    assert params["side"] == ["BUY"]
    assert params["symbol"] == ["BTCUSDT"]
    assert params["reduceOnly"] == ["false"]
    assert params["recvWindow"] == ["60000"]
    assert float(params["price"][0]) == pytest.approx(intent.limit_price)
    assert params["price"][0] == format_binance_decimal(intent.limit_price)
    assert float(params["quantity"][0]) == pytest.approx(intent.qty)
    assert "signature" in params
    assert "fapi.binance.com" not in req.full_url


def test_testnet_limit_without_price_does_not_post(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_urlopen(monkeypatch)
    broker = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    result = broker.submit(_intent(limit_price=None, entry_type="LIMIT_POST_ONLY"))
    assert result.status == "rejected"
    assert [c["req"].get_method() for c in captured if c["req"].get_method() == "POST"] == []
    assert "limit_price" in result.detail["error"]


def test_round_to_step_floors_qty() -> None:
    assert round_to_step(6277.256807168003, "1") == 6277.0
    assert round_to_step(2770.6555310271797, "0.1") == pytest.approx(2770.6)
    parsed = parse_symbol_filters(
        {
            "symbols": [
                {
                    "symbol": "AKEUSDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "1"},
                        {"filterType": "PRICE_FILTER", "tickSize": "0.000001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    ],
                }
            ]
        }
    )
    assert parsed["AKEUSDT"]["stepSize"] == "1"


def test_parse_usdt_wallet_from_balance_and_account() -> None:
    bal = parse_usdt_wallet(
        [{"asset": "BTC", "balance": "1"}, {"asset": "USDT", "balance": "3044.70813027", "availableBalance": "3000"}]
    )
    assert bal["equity_usdt"] == pytest.approx(3044.70813027)
    assert bal["available_usdt"] == pytest.approx(3000)
    acc = parse_usdt_wallet(
        {
            "totalWalletBalance": "3044.7",
            "availableBalance": "3010",
            "positions": [
                {"symbol": "BTCUSDT", "positionAmt": "0.0", "entryPrice": "0", "unrealizedProfit": "0"},
                {"symbol": "ETHUSDT", "positionAmt": "0.5", "entryPrice": "2000", "unrealizedProfit": "10"},
            ],
        }
    )
    assert acc["equity_usdt"] == pytest.approx(3044.7)
    assert acc["open_positions"] == 1
    assert acc["positions"][0]["symbol"] == "ETHUSDT"
    assert acc["positions"][0]["side"] == "LONG"


def test_fetch_wallet_reads_usdt(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float | None = None, context: Any = None) -> _FakeResponse:
        path = urllib.parse.urlparse(req.full_url).path
        if path.endswith("/time"):
            return _FakeResponse({"serverTime": 1_000_000_000_000})
        if path.endswith("/account"):
            return _FakeResponse(
                {"totalWalletBalance": "3044.70", "availableBalance": "3044.70", "positions": []}
            )
        return _FakeResponse({"error": path})

    monkeypatch.setattr("jev_trader.execution.urllib.request.urlopen", fake_urlopen)
    broker = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    wallet = broker.fetch_wallet()
    assert wallet["equity_usdt"] == pytest.approx(3044.70)
    assert wallet["available_usdt"] == pytest.approx(3044.70)
    assert wallet["open_positions"] == 0


def test_open_sell_is_rejected_on_paper_and_testnet(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_urlopen(monkeypatch)
    open_short = _intent(action="sell_short", order_side="SELL", reduce_only=False)
    paper = PaperBroker().submit(open_short)
    assert paper.status == "rejected"
    assert paper.detail["error"] == "sell_only_closes_long"
    broker = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    result = broker.submit(open_short)
    assert result.status == "rejected"
    assert result.detail["error"] == "sell_only_closes_long"
    assert captured == []


def test_testnet_market_flatten_omits_price(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_urlopen(monkeypatch)
    broker = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    intent = _intent(
        action="close",
        entry_type="MARKET",
        reduce_only=True,
        order_side="SELL",
        limit_price=None,
        qty=0.25,
    )
    result = broker.submit(intent)
    assert result.status == "filled"
    assert is_real_fill(result)
    posts = [c["req"] for c in captured if c["req"].get_method() == "POST"]
    params = urllib.parse.parse_qs(urllib.parse.urlparse(posts[0].full_url).query)
    assert params["type"] == ["MARKET"]
    assert params["reduceOnly"] == ["true"]
    assert "price" not in params
    assert "timeInForce" not in params


def test_new_gtx_body_is_not_a_real_fill() -> None:
    fake = ExecutionResult(
        status="accepted",
        venue="binance_testnet",
        client_order_id="x",
        reduce_only=False,
        detail={"http_status": 200, "body": {"status": "NEW", "executedQty": "0", "orderId": 9}},
    )
    assert is_real_fill(fake) is False
    filled = ExecutionResult(
        status="filled",
        venue="binance_testnet",
        client_order_id="x",
        reduce_only=False,
        detail={
            "http_status": 200,
            "body": {"status": "FILLED", "executedQty": "10", "avgPrice": "1.5"},
        },
    )
    assert is_real_fill(filled) is True


def test_flatten_all_market_closes_wallet_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_urlopen(monkeypatch)

    def fake_urlopen(req: Any, timeout: float | None = None, context: Any = None) -> _FakeResponse:
        captured.append({"req": req, "timeout": timeout, "context": context})
        parsed = urllib.parse.urlparse(req.full_url)
        path = parsed.path
        method = req.get_method()
        params = urllib.parse.parse_qs(parsed.query)
        if path.endswith("/time"):
            return _FakeResponse({"serverTime": 1_000_000_000_000})
        if path.endswith("/account") or path.endswith("/balance"):
            return _FakeResponse(
                {
                    "totalWalletBalance": "2800",
                    "availableBalance": "2000",
                    "positions": [
                        {
                            "symbol": "ETHUSDT",
                            "positionAmt": "0.5",
                            "entryPrice": "2000",
                            "unrealizedProfit": "-10",
                        }
                    ],
                }
            )
        if path.endswith("/openOrders"):
            return _FakeResponse([])
        if path.endswith("/exchangeInfo"):
            return _FakeResponse({"symbols": []})
        if method == "DELETE":
            return _FakeResponse({"code": 200})
        if method == "POST" and path.endswith("/order"):
            return _FakeResponse(
                {
                    "orderId": 99,
                    "status": "FILLED",
                    "executedQty": (params.get("quantity") or ["0.5"])[0],
                    "avgPrice": "1990",
                    "symbol": "ETHUSDT",
                }
            )
        return _FakeResponse()

    monkeypatch.setattr("jev_trader.execution.urllib.request.urlopen", fake_urlopen)
    broker = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    result = broker.flatten_all()
    posts = [c["req"] for c in captured if c["req"].get_method() == "POST"]
    assert posts
    params = urllib.parse.parse_qs(urllib.parse.urlparse(posts[0].full_url).query)
    assert params["symbol"] == ["ETHUSDT"]
    assert params["type"] == ["MARKET"]
    assert params["reduceOnly"] == ["true"]
    assert params["side"] == ["SELL"]
    assert result["closes"][0]["status"] == "filled"
