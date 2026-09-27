"""A reduce-only close keeps the closePosition stop until it is confirmed filled."""
from __future__ import annotations

from typing import Any

from jev_trader.execution import BinanceTestnetBroker
from jev_trader.models import market_close_intent

SYM = "BTCUSDT"
STOP = {"orderId": 7, "type": "STOP_MARKET", "side": "SELL", "closePosition": True, "stopPrice": "90"}


def _broker(post_answer: tuple[int, Any], *, query_answer: tuple[int, Any] = (400, {"code": -2013, "msg": "Order does not exist."})):
    broker = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    broker._time_synced = True
    broker._fill_poll_attempts = 1
    broker._fill_poll_sleep = 0
    broker.filters_for = lambda symbol: {"stepSize": "0.001"}  # type: ignore[method-assign]
    broker._live_position = lambda symbol: {"symbol": SYM, "side": "LONG", "size": 1.0}  # type: ignore[method-assign]
    calls: list[tuple[str, str]] = []

    def fake_request(method: str, path: str, params: dict | None = None, signed: bool = False, timeout: float = 10.0, **_kw: Any):
        calls.append((method, path))
        if method == "GET" and path == "/fapi/v1/openOrders":
            return 200, [dict(STOP)]
        if method == "POST" and path == "/fapi/v1/order":
            return post_answer
        if method == "GET" and path == "/fapi/v1/order":
            return query_answer
        return 200, {}

    broker._request = fake_request  # type: ignore[method-assign]
    return broker, calls


def _close():
    return market_close_intent(symbol=SYM, qty=1.0, order_side="SELL", client_order_id="jc_1", risk_event=None)


def _stop_cancelled(calls) -> bool:
    return ("DELETE", "/fapi/v1/allOpenOrders") in calls or ("DELETE", "/fapi/v1/order") in calls


def test_rejected_close_leaves_the_stop() -> None:
    broker, calls = _broker((400, {"code": -1111, "msg": "Precision is over the maximum defined for this asset."}))
    result = broker.submit(_close())
    assert result.status == "rejected"
    assert ("POST", "/fapi/v1/order") in calls
    assert not _stop_cancelled(calls)


def test_unknown_close_leaves_the_stop() -> None:
    broker, calls = _broker((503, {"error": "Service Unavailable"}))
    result = broker.submit(_close())
    assert result.status == "submit_unknown"
    assert not _stop_cancelled(calls)


def test_partial_close_keeps_the_stop() -> None:
    body = {"orderId": 9, "status": "PARTIALLY_FILLED", "executedQty": "0.4", "avgPrice": "100"}
    broker, calls = _broker((200, body), query_answer=(200, body))
    result = broker.submit(_close())
    assert result.status == "filled"
    assert result.detail["stop_kept"] == "partial_close"
    assert not _stop_cancelled(calls)


def test_full_close_clears_the_stop_after_the_fill() -> None:
    broker, calls = _broker((200, {"orderId": 9, "status": "FILLED", "executedQty": "1", "avgPrice": "100"}))
    result = broker.submit(_close())
    assert result.status == "filled" and result.detail["stop_cleared"] is True
    post_i = calls.index(("POST", "/fapi/v1/order"))
    assert ("DELETE", "/fapi/v1/allOpenOrders") in calls[post_i:]
    assert ("DELETE", "/fapi/v1/allOpenOrders") not in calls[:post_i]


def test_flatten_keeps_stop_of_open_position_until_close_fills() -> None:
    broker = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    broker._time_synced = True
    broker._fill_poll_attempts = 1
    broker._fill_poll_sleep = 0
    broker.filters_for = lambda symbol: {"stepSize": "0.001"}  # type: ignore[method-assign]
    wallet = {"equity_usdt": 1000.0, "positions": [{"symbol": SYM, "side": "LONG", "size": 1.0}], "positions_known": True}
    broker.fetch_wallet = lambda ttl=0: wallet  # type: ignore[method-assign]
    calls: list[tuple[str, str, dict]] = []

    def fake_request(method: str, path: str, params: dict | None = None, signed: bool = False, timeout: float = 10.0, **_kw: Any):
        calls.append((method, path, dict(params or {})))
        if method == "GET" and path == "/fapi/v1/openOrders":
            rows = [dict(STOP, symbol=SYM), {"orderId": 8, "symbol": "ETHUSDT", "type": "LIMIT", "timeInForce": "GTX"}]
            return 200, [r for r in rows if not (params or {}).get("symbol") or r["symbol"] == params["symbol"]]
        if method == "POST":
            return 400, {"code": -1111, "msg": "rejected"}
        return 200, {}

    broker._request = fake_request  # type: ignore[method-assign]
    out = broker.flatten_all()
    assert out["closes"][0]["status"] == "rejected"
    wholesale = [c[2].get("symbol") for c in calls if c[0] == "DELETE" and c[1] == "/fapi/v1/allOpenOrders"]
    assert wholesale == ["ETHUSDT"]  # flat symbol cleaned; BTC stop survived the rejected close
