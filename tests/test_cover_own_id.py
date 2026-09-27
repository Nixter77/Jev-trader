"""Short cover sent as plain BUY: own jf… id in the ledger, one in flight at a time."""
from __future__ import annotations

from typing import Any

from jev_trader.execution import BinanceTestnetBroker
from jev_trader.ledger import Ledger
from jev_trader.models import CycleResult, market_close_intent

SYM = "BTCUSDT"
NOT_FOUND = (400, {"code": -2013, "msg": "Order does not exist."})


class Exchange:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.post_answers: list[tuple[int, Any]] = []
        self.orders: dict[str, tuple[int, Any]] = {}

    def request(self, method: str, path: str, params: dict | None = None, signed: bool = False, timeout: float = 10.0, **_kw: Any):
        params = dict(params or {})
        self.calls.append((method, path, params))
        if method == "POST" and path == "/fapi/v1/order":
            return self.post_answers.pop(0)
        if method == "GET" and path == "/fapi/v1/order":
            return self.orders.get(str(params.get("origClientOrderId")), NOT_FOUND)
        if method == "GET" and path == "/fapi/v1/openOrders":
            return 200, []
        return 200, {}


def _broker(ex: Exchange) -> BinanceTestnetBroker:
    b = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    b._time_synced = True
    b._fill_poll_attempts = 1
    b._fill_poll_sleep = 0
    b.filters_for = lambda symbol: {"stepSize": "0.001"}  # type: ignore[method-assign]
    b._live_position = lambda symbol: {"symbol": SYM, "side": "SHORT", "size": 1.0}  # type: ignore[method-assign]
    b._request = ex.request  # type: ignore[method-assign]
    return b


REDUCE_ONLY_REJECTED = (400, {"code": -2022, "msg": "ReduceOnly Order is rejected."})


def _cover(cid: str = "jc_s1"):
    return market_close_intent(symbol=SYM, qty=1.0, order_side="BUY", client_order_id=cid, risk_event=None)


def _record(ledger: Ledger, intent, execution) -> None:
    ledger.record(CycleResult(action="close", skip_reason=None, intent=intent, execution=execution, judgment=None, state_text="", state={"symbol": SYM}))


def _posts(ex: Exchange) -> list[dict]:
    return [p for m, path, p in ex.calls if m == "POST"]


def test_unknown_cover_is_recorded_under_its_own_id(tmp_path) -> None:
    ex = Exchange()
    ex.post_answers = [REDUCE_ONLY_REJECTED, (503, {"error": "Service Unavailable"})]
    broker = _broker(ex)
    intent = _cover()
    result = broker.submit(intent)
    assert result.status == "submit_unknown"
    cover_cid = _posts(ex)[1]["newClientOrderId"]
    assert result.client_order_id == cover_cid != intent.client_order_id
    ledger = Ledger(tmp_path / "l.sqlite")
    _record(ledger, intent, result)
    with ledger._connect() as conn:
        cids = [r[0] for r in conn.execute("SELECT client_order_id FROM orders")]
    assert cids == [cover_cid]
    assert ledger.unknown_cover_cids(SYM) == [cover_cid]


def test_filled_cover_fill_uses_its_own_id(tmp_path) -> None:
    ex = Exchange()
    ex.post_answers = [REDUCE_ONLY_REJECTED, (200, {"orderId": 5, "status": "FILLED", "executedQty": "1", "avgPrice": "99"})]
    broker = _broker(ex)
    intent = _cover()
    result = broker.submit(intent)
    ledger = Ledger(tmp_path / "l.sqlite")
    _record(ledger, intent, result)
    fills = ledger.book()["fills"]
    assert [f["client_order_id"] for f in fills] == [result.client_order_id]


def test_second_cover_waits_while_first_is_unknown() -> None:
    ex = Exchange()
    ex.post_answers = [REDUCE_ONLY_REJECTED, (503, {"error": "down"}), REDUCE_ONLY_REJECTED]
    broker = _broker(ex)
    first = broker.submit(_cover("jc_s1"))
    assert first.status == "submit_unknown"
    second = broker.submit(_cover("jc_s2"))
    assert second.status == "rejected" and second.detail["error"] == "cover_submit_unknown"
    assert second.detail["pending_client_order_id"] == first.client_order_id
    # Only the reduce-only tries and the first cover went out.
    assert [p.get("reduceOnly", "plain") for p in _posts(ex)] == ["true", "plain", "true"]


def test_cover_goes_once_the_old_one_is_final() -> None:
    ex = Exchange()
    ex.post_answers = [REDUCE_ONLY_REJECTED, (503, {"error": "down"})]
    broker = _broker(ex)
    first = broker.submit(_cover("jc_s1"))
    ex.orders[first.client_order_id] = (200, {"orderId": 5, "status": "CANCELED", "executedQty": "0"})
    ex.post_answers = [REDUCE_ONLY_REJECTED, (200, {"orderId": 6, "status": "FILLED", "executedQty": "1", "avgPrice": "99"})]
    second = broker.submit(_cover("jc_s2"))
    assert second.status == "filled"


def test_ledger_row_blocks_cover_after_restart() -> None:
    ex = Exchange()
    ex.post_answers = [REDUCE_ONLY_REJECTED]
    broker = _broker(ex)
    broker.unknown_cover_cids_fn = lambda symbol: ["jf_old"]
    result = broker.submit(_cover())
    assert result.detail["error"] == "cover_submit_unknown"
    assert result.detail["pending_client_order_id"] == "jf_old"


def test_unknown_cover_released_after_grace() -> None:
    ex = Exchange()
    ex.post_answers = [REDUCE_ONLY_REJECTED, (503, {"error": "down"})]
    broker = _broker(ex)
    first = broker.submit(_cover("jc_s1"))
    cid, _ = broker._unknown_covers[SYM]
    broker._unknown_covers[SYM] = (cid, -1e9)  # long ago
    ex.post_answers = [REDUCE_ONLY_REJECTED, (200, {"orderId": 6, "status": "FILLED", "executedQty": "1", "avgPrice": "99"})]
    second = broker.submit(_cover("jc_s2"))
    assert second.status == "filled"
    assert any(m == "DELETE" and p.get("origClientOrderId") == first.client_order_id for m, _path, p in ex.calls)
