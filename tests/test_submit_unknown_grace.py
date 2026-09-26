"""Item 5: submit_unknown survives quick -2013, counts in the cap, settles after grace."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jev_trader.ledger import Ledger
from jev_trader.models import CycleResult, ExecutionResult, TradeIntent
from jev_trader.reconcile import Reconciler

from tests.test_reconcile import SYM, FakeBroker, _fills


def _record_unknown(ledger: Ledger, cid: str) -> None:
    intent = TradeIntent(
        action="buy_long",
        qty=1.0,
        stop_price=95.0,
        stop_distance=5.0,
        entry_type="LIMIT_POST_ONLY",
        reduce_only=False,
        client_order_id=cid,
        symbol=SYM,
        risk_pct=0.005,
        limit_price=100.0,
    )
    ledger.record(
        CycleResult(
            action="buy_long",
            skip_reason=None,
            intent=intent,
            execution=ExecutionResult(
                status="submit_unknown",
                venue="binance_testnet",
                client_order_id=cid,
                reduce_only=False,
                detail={"http_status": 0, "body": {"error": "timed out"}, "error": "submit_unknown"},
            ),
            judgment=None,
            state_text="",
            state={"symbol": SYM},
        )
    )


class LaggyBroker(FakeBroker):
    """Order unknown to GET; cancel may reveal it."""

    def __init__(self) -> None:
        super().__init__()
        self.cancel_body: dict | None = None

    def cancel_order(self, symbol, cid):
        self.cancelled.append(cid)
        if self.cancel_body is None:
            return {"http_status": 400, "body": {"code": -2011, "msg": "Unknown order sent."}}
        return {"http_status": 200, "body": dict(self.cancel_body)}


def _rec(ledger, broker, ahead: float) -> Reconciler:
    return Reconciler(broker, ledger, min_interval_sec=0, now_fn=lambda: datetime.now(timezone.utc) + timedelta(seconds=ahead))


def _hour_ago() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=1)


def test_young_submit_unknown_not_found_stays_pending_and_counted(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = LaggyBroker()
    _record_unknown(ledger, "jev1_u")
    summary = _rec(ledger, broker, ahead=30).run_once()
    assert [w["cid"] for w in summary["unknown_waiting"]] == ["jev1_u"]
    assert broker.cancelled == []
    assert len(ledger.pending_orders(_hour_ago())) == 1
    assert ledger.count_entries_since(_hour_ago()) == 1


def test_submit_unknown_after_grace_is_cancelled_and_settled(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = LaggyBroker()
    _record_unknown(ledger, "jev1_u")
    summary = _rec(ledger, broker, ahead=301).run_once()
    assert broker.cancelled == ["jev1_u"]
    assert summary["final"][0]["status"] == "not_found"
    assert ledger.pending_orders(_hour_ago() - timedelta(minutes=10)) == []
    assert ledger.count_entries_since(_hour_ago()) == 0


def test_submit_unknown_surfacing_on_cancel_records_its_fill(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = LaggyBroker()
    _record_unknown(ledger, "jev1_u")
    broker.cancel_body = {"orderId": 55, "status": "CANCELED", "executedQty": "0.4", "price": "100", "avgPrice": "100"}
    broker.trade(55, "BUY", 0.4, 100.0, datetime.now(timezone.utc))
    broker.long(0.4)
    _rec(ledger, broker, ahead=301).run_once()
    fills = _fills(ledger)
    assert [(f["client_order_id"], f["qty"]) for f in fills] == [("jev1_u", 0.4)]
    assert ledger.pending_orders(_hour_ago() - timedelta(minutes=10)) == []


def test_working_order_not_found_still_settles_at_once(tmp_path) -> None:
    from tests.test_reconcile import _record_entry

    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    del broker.orders["jev1_a"]
    Reconciler(broker, ledger, min_interval_sec=0).run_once()
    assert ledger.pending_orders(_hour_ago()) == []
