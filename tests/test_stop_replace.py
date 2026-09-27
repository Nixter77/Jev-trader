"""ensure_stop_market checks the stop price, not only that a stop exists."""
from __future__ import annotations

from typing import Any

from jev_trader.execution import BinanceTestnetBroker


class Exchange:
    def __init__(self, stops: list[dict[str, Any]], *, refuse: list[tuple[int, Any]] | None = None) -> None:
        self.open = [dict(r) for r in stops]
        self.calls: list[tuple[str, str, dict]] = []
        self.refuse = list(refuse or [])
        self.next_id = 100

    def request(self, method: str, path: str, params: dict | None = None, signed: bool = False, timeout: float = 10.0, **_kw: Any):
        params = dict(params or {})
        self.calls.append((method, path, params))
        if method == "GET" and path == "/fapi/v1/openOrders":
            return 200, [dict(r) for r in self.open]
        if method == "DELETE" and path == "/fapi/v1/order":
            self.open = [r for r in self.open if r.get("orderId") != params.get("orderId")]
            return 200, {"status": "CANCELED"}
        if method == "POST" and path == "/fapi/v1/order":
            if self.refuse:
                return self.refuse.pop(0)
            if any(r.get("closePosition") for r in self.open):
                return 400, {"code": -4130, "msg": "closePosition stop exists"}
            self.next_id += 1
            self.open.append(
                {"orderId": self.next_id, "type": "STOP_MARKET", "side": params["side"], "closePosition": True, "stopPrice": params["stopPrice"]}
            )
            return 200, {"orderId": self.next_id, "status": "NEW"}
        return 200, {}


def _broker(ex: Exchange) -> BinanceTestnetBroker:
    b = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    b.filters_for = lambda symbol: {"tickSize": "0.1"}  # type: ignore[method-assign]
    b._request = ex.request  # type: ignore[method-assign]
    return b


def _stop(oid: int, price: str) -> dict[str, Any]:
    return {"orderId": oid, "type": "STOP_MARKET", "side": "SELL", "closePosition": True, "stopPrice": price}


def _posts(ex: Exchange) -> list[dict]:
    return [p for m, path, p in ex.calls if m == "POST"]


def test_same_price_stop_is_kept() -> None:
    ex = Exchange([_stop(1, "95.00")])
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.04)  # rounds to 95.0
    assert res["body"]["skipped"] == "stop_exists"
    assert _posts(ex) == []


def test_stale_stop_from_old_position_is_replaced() -> None:
    ex = Exchange([_stop(1, "80.0")])
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["http_status"] == 200
    assert res["replaced"]["old_stop_prices"] == ["80.0"]
    assert [r["stopPrice"] for r in ex.open] == ["95"]
    # Cancel went before the new stop (Binance allows one per direction).
    kinds = [(m, path) for m, path, _ in ex.calls if m in {"POST", "DELETE"}]
    assert kinds == [("DELETE", "/fapi/v1/order"), ("POST", "/fapi/v1/order")]


def test_refused_new_stop_puts_old_one_back() -> None:
    ex = Exchange([_stop(1, "80.0")], refuse=[(400, {"code": -2021, "msg": "Order would immediately trigger."})])
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["body"]["code"] == -2021  # callers fail-close on this
    assert res["restored_old_stop"]["http_status"] == 200
    assert [r["stopPrice"] for r in ex.open] == ["80"]


def test_no_stop_places_one() -> None:
    ex = Exchange([])
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["http_status"] == 200
    assert len(ex.open) == 1


def test_stop_without_price_field_is_not_churned() -> None:
    ex = Exchange([{"orderId": 1, "type": "STOP_MARKET", "side": "SELL", "closePosition": True}])
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["body"]["skipped"] == "stop_exists"


def test_unknown_tick_size_does_not_churn_the_stop() -> None:
    # exchangeInfo unavailable: no tick to round to. A replace would cancel the
    # good stop, send an unrounded price (-1111) and restore, every cycle.
    ex = Exchange([_stop(1, "95.0")], refuse=[(400, {"code": -1111, "msg": "Precision is over the maximum defined for this asset."})] * 3)
    broker = _broker(ex)
    broker.filters_for = lambda symbol: {}  # type: ignore[method-assign]
    for _ in range(3):
        res = broker.ensure_stop_market("BTCUSDT", stop_price=95.0433)
        assert res["body"] == {"skipped": "stop_exists", "unverified": "no_tick_size"}
    assert [c for c in ex.calls if c[0] in {"DELETE", "POST"}] == []
    assert [r["stopPrice"] for r in ex.open] == ["95.0"]
