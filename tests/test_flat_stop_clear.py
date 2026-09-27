"""After a blind / unknown close fills, the old stop is cleared once the symbol is verified flat."""
from __future__ import annotations

from typing import Any

from jev_trader.cycle import CycleResult
from jev_trader.execution import TEMP_STOP_PREFIX, ExecutionResult, PositionsUnknown
from jev_trader.ledger import Ledger
from jev_trader.models import market_close_intent
from tests.test_reconcile import SYM, FakeBroker, _rec
from tests.test_stop_swap_covered import Exchange, _broker

CLOSE_STOP = {"orderId": 7, "type": "STOP_MARKET", "side": "SELL", "stopPrice": "95", "closePosition": True, "clientOrderId": "js_old"}
TEMP_STOP = {"orderId": 8, "type": "STOP_MARKET", "side": "SELL", "stopPrice": "95", "reduceOnly": True, "clientOrderId": f"{TEMP_STOP_PREFIX}_x"}


class CloseVenue(Exchange):
    """Exchange fake that also fills a reduce-only MARKET close."""

    def request(self, method: str, path: str, params: dict | None = None, signed: bool = False, timeout: float = 10.0, **kw: Any):
        p = dict(params or {})
        if method == "POST" and path == "/fapi/v1/order" and p.get("type") == "MARKET":
            self.calls.append((method, path, p))
            return 200, {"orderId": 91, "status": "FILLED", "executedQty": p["quantity"], "avgPrice": "95"}
        if method == "DELETE" and path in {"/fapi/v1/allOpenOrders", "/fapi/v1/openOrders"}:
            self.calls.append((method, path, p))
            return 200, {}
        return super().request(method, path, params, signed, timeout, **kw)


def _blind_broker(ex: Exchange, after: list[Any]):
    """First position read is unknown (forces the blind path); later reads follow `after`."""
    b = _broker(ex)
    b._time_synced = True
    b._fill_poll_sleep = 0
    reads = iter(["unknown", *after])

    def live(symbol: str):
        nxt = next(reads, after[-1])
        if nxt == "unknown":
            raise PositionsUnknown("balance only")
        return nxt

    b._live_position = live  # type: ignore[method-assign]
    return b


def _kill(cid: str = "jc_k"):
    return market_close_intent(symbol=SYM, qty=1.0, order_side="SELL", client_order_id=cid, risk_event="kill_switch")


def test_blind_close_filled_and_verified_flat_clears_old_stops() -> None:
    ex = CloseVenue([CLOSE_STOP, TEMP_STOP])
    b = _blind_broker(ex, [None])
    res = b.submit(_kill())
    assert res.status == "filled" and res.detail["blind_close"]
    assert res.detail["flat_check"]["cleared"] == 2
    assert res.detail["stop_cleared"] is True
    assert ex.open == []
    assert SYM not in b.stop_sweep_pending


def test_blind_close_with_positions_still_unknown_keeps_stop_and_queues_symbol() -> None:
    ex = CloseVenue([CLOSE_STOP])
    b = _blind_broker(ex, ["unknown"])
    res = b.submit(_kill())
    assert res.status == "filled"
    assert res.detail["flat_check"] == {"kept": "positions_unknown"}
    assert [r["orderId"] for r in ex.open] == [7]
    assert b.stop_sweep_pending == {SYM}
    # Positions readable again and flat: the queued sweep clears it.
    b._live_position = lambda symbol: None  # type: ignore[method-assign]
    assert b.clear_stops_if_flat(SYM)["cleared"] == 1
    assert ex.open == [] and b.stop_sweep_pending == set()


def test_blind_close_leaving_a_position_keeps_its_stop() -> None:
    ex = CloseVenue([CLOSE_STOP])
    b = _blind_broker(ex, [{"symbol": SYM, "side": "LONG", "size": 0.3}])
    res = b.submit(_kill())
    assert res.status == "filled"
    assert res.detail["flat_check"] == {"kept": "position"}
    assert "stop_cleared" not in res.detail
    assert [r["orderId"] for r in ex.open] == [7]


class SweepBroker(FakeBroker):
    def __init__(self) -> None:
        super().__init__()
        self.flat_checks: list[str] = []
        self.stop_sweep_pending: set[str] = set()
        self.answer: dict[str, Any] = {"cleared": 1, "cancels": [{"http_status": 200}]}

    def clear_stops_if_flat(self, symbol: str) -> dict[str, Any]:
        self.flat_checks.append(symbol)
        if self.answer.get("cleared"):
            self.stop_sweep_pending.discard(symbol)
        return self.answer


def _unknown_close(ledger: Ledger, broker: SweepBroker, cid: str) -> None:
    intent = market_close_intent(symbol=SYM, qty=1.0, order_side="SELL", client_order_id=cid, risk_event="stop")
    ledger.record(
        CycleResult(
            action="close",
            skip_reason=None,
            intent=intent,
            execution=ExecutionResult(status="submit_unknown", venue="binance_testnet", client_order_id=cid, reduce_only=True, detail={"http_status": 503}),
            judgment=None,
            state_text="",
            state={"symbol": SYM},
        )
    )


def test_unknown_close_resolved_filled_clears_stop_on_flat_symbol(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = SweepBroker()
    broker.flat()
    _unknown_close(ledger, broker, "jc_u1")
    broker.orders["jc_u1"] = {"orderId": 55, "status": "FILLED", "executedQty": "1", "avgPrice": "94", "price": "0"}
    summary = _rec(ledger, broker).run_once()
    assert broker.flat_checks == [SYM]
    assert {"symbol": SYM, "flat_cleared": 1} in summary["stops"]


def test_unknown_close_that_did_not_fill_does_not_touch_stops(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = SweepBroker()
    broker.long(1.0)
    _unknown_close(ledger, broker, "jc_u2")
    broker.orders["jc_u2"] = {"orderId": 56, "status": "EXPIRED", "executedQty": "0", "avgPrice": "0", "price": "0"}
    _rec(ledger, broker).run_once()
    assert broker.flat_checks == []


def test_reconcile_pass_retries_queued_flat_checks(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = SweepBroker()
    broker.stop_sweep_pending = {SYM}
    broker.answer = {"kept": "positions_unknown"}
    _rec(ledger, broker).run_once()
    assert broker.flat_checks == [SYM] and broker.stop_sweep_pending == {SYM}
    broker.answer = {"cleared": 1}
    _rec(ledger, broker).run_once()
    assert broker.flat_checks == [SYM, SYM] and broker.stop_sweep_pending == set()
