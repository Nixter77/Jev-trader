"""A closePosition stop swap is covered by a temporary reduce-only stop; a bare long is closed."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from jev_trader.cycle import STOP_UNPROTECTED, run_once
from jev_trader.execution import TEMP_STOP_PREFIX, BinanceTestnetBroker, PaperBroker
from jev_trader.reconcile import _must_fail_close


class Exchange:
    """openOrders + POST/DELETE order. One closePosition stop per side (-4130)."""

    def __init__(self, stops: list[dict[str, Any]], *, refuse: dict[str, list[tuple[int, Any]]] | None = None) -> None:
        self.open = [dict(r) for r in stops]
        self.calls: list[tuple[str, str, dict]] = []
        self.refuse = {k: list(v) for k, v in (refuse or {}).items()}
        self.next_id = 100
        self.bare_moments = 0  # write calls after which no SELL stop was open

    def _kind(self, params: dict) -> str:
        return "close" if params.get("closePosition") == "true" else "temp"

    def request(self, method: str, path: str, params: dict | None = None, signed: bool = False, timeout: float = 10.0, **_kw: Any):
        params = dict(params or {})
        self.calls.append((method, path, params))
        out: tuple[int, Any] = (200, {})
        if method == "GET" and path == "/fapi/v1/openOrders":
            return 200, [dict(r) for r in self.open]
        if method == "DELETE" and path == "/fapi/v1/order":
            before = len(self.open)
            self.open = [
                r for r in self.open
                if not (
                    (params.get("orderId") is not None and r.get("orderId") == params.get("orderId"))
                    or (params.get("origClientOrderId") and r.get("clientOrderId") == params.get("origClientOrderId"))
                )
            ]
            out = (200, {"status": "CANCELED"}) if len(self.open) < before else (400, {"code": -2011})
        elif method == "POST" and path == "/fapi/v1/order":
            kind = self._kind(params)
            if self.refuse.get(kind):
                out = self.refuse[kind].pop(0)
            elif kind == "close" and any(r.get("closePosition") for r in self.open):
                out = (400, {"code": -4130, "msg": "closePosition stop exists"})
            else:
                self.next_id += 1
                row = {
                    "orderId": self.next_id,
                    "type": "STOP_MARKET",
                    "side": params["side"],
                    "stopPrice": params["stopPrice"],
                    "clientOrderId": params.get("newClientOrderId") or f"web_{self.next_id}",
                }
                if kind == "close":
                    row["closePosition"] = True
                else:
                    row["reduceOnly"] = True
                    row["origQty"] = params["quantity"]
                self.open.append(row)
                out = (200, {"orderId": self.next_id, "status": "NEW"})
        if method in {"POST", "DELETE"} and not any(r.get("side") == "SELL" for r in self.open):
            self.bare_moments += 1
        return out


def _broker(ex: Exchange, *, size: float | None = 0.5) -> BinanceTestnetBroker:
    b = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    b.filters_for = lambda symbol: {"tickSize": "0.1", "stepSize": "0.001"}  # type: ignore[method-assign]
    b._request = ex.request  # type: ignore[method-assign]
    b.invalidate_wallet = lambda: None  # type: ignore[method-assign]
    b._live_position = (  # type: ignore[method-assign]
        lambda symbol: None if size is None else {"symbol": symbol, "side": "LONG", "size": size}
    )
    return b


def _stop(oid: int, price: str) -> dict[str, Any]:
    return {"orderId": oid, "type": "STOP_MARKET", "side": "SELL", "closePosition": True, "stopPrice": price, "clientOrderId": f"old{oid}"}


def _writes(ex: Exchange) -> list[tuple[str, dict]]:
    return [(m, p) for m, path, p in ex.calls if m in {"POST", "DELETE"}]


def test_swap_places_temp_stop_first_and_removes_it_after() -> None:
    ex = Exchange([_stop(1, "80.0")])
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["http_status"] == 200
    writes = _writes(ex)
    assert [m for m, _ in writes] == ["POST", "DELETE", "POST", "DELETE"]
    temp = writes[0][1]
    assert temp["reduceOnly"] == "true" and temp["quantity"] == "0.5"
    assert temp["stopPrice"] == "95" and "closePosition" not in temp
    assert temp["newClientOrderId"].startswith(TEMP_STOP_PREFIX)
    assert writes[1][1].get("orderId") == 1  # old closePosition stop
    assert writes[2][1]["closePosition"] == "true"
    assert writes[3][1]["origClientOrderId"] == temp["newClientOrderId"]
    assert ex.bare_moments == 0  # never without a SELL stop
    assert [(r["stopPrice"], bool(r.get("closePosition"))) for r in ex.open] == [("95", True)]


def test_refused_new_stop_keeps_temp_as_protection_then_next_pass_finishes() -> None:
    ex = Exchange([_stop(1, "80.0")], refuse={"close": [(400, {"code": -1001, "msg": "internal"})]})
    broker = _broker(ex)
    res = broker.ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["protected_by"] == "temp_stop"
    assert "restored_old_stop" not in res and not res.get("unprotected")
    assert not _must_fail_close(res)
    assert [(r["stopPrice"], bool(r.get("reduceOnly"))) for r in ex.open] == [("95", True)]
    assert ex.bare_moments == 0
    again = broker.ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert again["http_status"] == 200
    assert [(r["stopPrice"], bool(r.get("closePosition"))) for r in ex.open] == [("95", True)]


def test_existing_temp_at_target_is_reused_not_duplicated() -> None:
    temp = {"orderId": 7, "type": "STOP_MARKET", "side": "SELL", "reduceOnly": True, "stopPrice": "95.0", "clientOrderId": "jtBTCUSDT1"}
    ex = Exchange([_stop(1, "80.0"), temp])
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["http_status"] == 200
    posts = [p for m, p in _writes(ex) if m == "POST"]
    assert len(posts) == 1 and posts[0]["closePosition"] == "true"
    assert [(r["stopPrice"], bool(r.get("closePosition"))) for r in ex.open] == [("95", True)]


def test_no_position_read_falls_back_and_flags_unprotected_when_restore_fails() -> None:
    refused = (400, {"code": -1001, "msg": "internal"})
    ex = Exchange([_stop(1, "80.0")], refuse={"close": [refused, refused]})
    res = _broker(ex, size=None).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["restored_old_stop"]["http_status"] == 400
    assert res["unprotected"] is True
    assert _must_fail_close(res)


def test_refused_temp_falls_back_to_restore() -> None:
    refused = (400, {"code": -2021, "msg": "Order would immediately trigger."})
    ex = Exchange([_stop(1, "80.0")], refuse={"temp": [refused], "close": [refused]})
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["restored_old_stop"]["http_status"] == 200
    assert not res.get("unprotected")
    assert _must_fail_close(res)  # -2021: price is through the stop
    assert [r["stopPrice"] for r in ex.open] == ["80"]


def test_cycle_closes_a_long_left_without_stop(market_snapshot) -> None:
    submitted: list[Any] = []

    class Broker(PaperBroker):
        def ensure_stop_market(self, symbol: str, *, stop_price: float, order_side: str = "SELL", **_kw: Any):
            return {"http_status": 400, "body": {"code": -1001}, "restored_old_stop": {"http_status": 400}, "unprotected": True}

        def submit(self, intent):
            submitted.append(intent)
            return super().submit(intent)

    class NoModel:
        def judge(self, _compact):
            raise AssertionError("model must not be called")

        def close(self) -> None:
            return None

    snapshot = replace(
        market_snapshot,
        position=replace(market_snapshot.position, side="LONG", size=0.01, entry=50_000.0, stop_price=40_000.0),
    )
    result = run_once(snapshot, jev_client=NoModel(), broker=Broker())
    assert len(submitted) == 1
    intent = submitted[0]
    assert intent.action == "close" and intent.reduce_only and intent.entry_type == "MARKET"
    assert intent.order_side == "SELL" and intent.risk_event == "stop"
    assert intent.qty == pytest.approx(0.01)
    assert result.action == "close"
    assert result.model_skipped


def test_cycle_holds_normally_when_stop_is_armed(market_snapshot) -> None:
    class Broker(PaperBroker):
        def ensure_stop_market(self, symbol: str, *, stop_price: float, order_side: str = "SELL", **_kw: Any):
            return {"http_status": 200, "body": {"skipped": "stop_exists"}}

    snapshot = replace(
        market_snapshot,
        position=replace(market_snapshot.position, side="LONG", size=0.01, entry=50_000.0, stop_price=40_000.0),
    )
    from jev_trader.jev import judgment_from_dict

    result = run_once(
        snapshot,
        judgment=judgment_from_dict(
            {"action": "hold", "trend_aligned": 0.2, "false_break_risk": 0.2, "signal_strength": "слабый", "should_trade_now": 0.1, "model": "jev-1.13.0"}
        ),
        broker=Broker(),
    )
    assert result.action == "hold"
    assert result.skip_reason != STOP_UNPROTECTED


def _filled_entry(tmp_path):
    from datetime import datetime, timezone

    from tests.test_reconcile import FakeBroker, _record_entry
    from jev_trader.ledger import Ledger

    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.orders["jev1_a"].update(status="FILLED", executedQty="1", avgPrice="100")
    broker.trade(11, "BUY", 1.0, 100.0, datetime.now(timezone.utc))
    broker.long(1.0)
    return ledger, broker


def test_reconcile_closes_when_swap_left_long_bare(tmp_path) -> None:
    from tests.test_reconcile import _rec

    ledger, broker = _filled_entry(tmp_path)
    broker.stop_answer = {"http_status": 400, "body": {"code": -1001}, "restored_old_stop": {"http_status": 400}, "unprotected": True}
    summary = _rec(ledger, broker).run_once()
    assert summary["stops"][0]["fail_closed"] == "filled"
    assert len(broker.submitted) == 1 and broker.submitted[0].reduce_only


def test_reconcile_does_not_close_when_temp_stop_protects(tmp_path) -> None:
    from tests.test_reconcile import _rec

    ledger, broker = _filled_entry(tmp_path)
    broker.stop_answer = {"http_status": 400, "body": {"code": -1001}, "protected_by": "temp_stop"}
    summary = _rec(ledger, broker).run_once()
    assert "fail_closed" not in summary["stops"][0]
    assert broker.submitted == []


class _LandsButTimesOut(Exchange):
    """The new closePosition stop reaches the book, but its answer is lost."""

    def request(self, method, path, params=None, signed=False, timeout=10.0, **kw):
        out = super().request(method, path, params, signed, timeout, **kw)
        p = dict(params or {})
        if method == "POST" and p.get("closePosition") == "true" and p.get("stopPrice") == "95" and out[0] == 200:
            return 0, {"error": "TimeoutError: timed out"}
        return out


def test_timed_out_new_stop_that_landed_is_not_called_unprotected() -> None:
    # No position read -> no temp. New stop times out but lands; the restore
    # then gets -4130. The long is protected: closing it would be spurious.
    ex = _LandsButTimesOut([_stop(1, "80.0")])
    res = _broker(ex, size=None).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["restored_old_stop"]["body"]["code"] == -4130
    assert not res.get("unprotected")
    assert not _must_fail_close(res)
    assert [r["stopPrice"] for r in ex.open] == ["95"]


class _NetworkDown(Exchange):
    def __init__(self, stops):
        super().__init__(stops)
        self.down = False

    def request(self, method, path, params=None, signed=False, timeout=10.0, **kw):
        if self.down:
            self.calls.append((method, path, dict(params or {})))
            return 0, {"error": "URLError: network unreachable"}
        out = super().request(method, path, params, signed, timeout, **kw)
        if method == "GET" and path == "/fapi/v1/openOrders":
            self.down = True  # drops right after the first listing
        return out


def test_network_outage_mid_swap_does_not_trigger_fail_close() -> None:
    ex = _NetworkDown([_stop(1, "80.0")])
    res = _broker(ex, size=None).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert [r["stopPrice"] for r in ex.open] == ["80.0"]  # the old stop never left
    assert not res.get("unprotected")
    assert res.get("protection_unknown") is True
    assert not _must_fail_close(res)
