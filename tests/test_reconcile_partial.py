"""Item 4: any executedQty > 0 on a resting entry arms the closePosition stop at once."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jev_trader.ledger import Ledger

from tests.test_reconcile import SYM, FakeBroker, _fills, _rec, _record_entry


def _pending(ledger):
    return ledger.pending_orders(datetime.now(timezone.utc) - timedelta(hours=1))


def test_young_partial_fill_arms_stop_once_and_stays_pending(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.orders["jev1_a"].update(status="PARTIALLY_FILLED", executedQty="0.3", avgPrice="100")
    broker.long(0.3)

    summary = _rec(ledger, broker).run_once()

    assert broker.stops == [(SYM, 95.0)]
    assert summary["stops"][0]["partial"] is True and summary["stops"][0]["armed"] is True
    assert broker.cancelled == []
    # Fill is recorded when the order is final, not now.
    assert _fills(ledger) == []
    assert len(_pending(ledger)) == 1

    _rec_same = _rec(ledger, broker)
    _rec_same._partial_armed = {"jev1_a"}
    _rec_same.run_once()
    assert broker.stops == [(SYM, 95.0)], "no re-arm every pass"


def test_partial_armed_once_per_reconciler(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.orders["jev1_a"].update(status="PARTIALLY_FILLED", executedQty="0.3", avgPrice="100")
    rec = _rec(ledger, broker)
    rec.run_once()
    rec.run_once()
    assert broker.stops == [(SYM, 95.0)]
    # Order then fills fully: recorded, stop not placed twice by us.
    broker.orders["jev1_a"].update(status="FILLED", executedQty="1", avgPrice="100")
    broker.trade(11, "BUY", 1.0, 100.0, datetime.now(timezone.utc))
    broker.long(1.0)
    rec.run_once()
    assert [f["client_order_id"] for f in _fills(ledger)] == ["jev1_a"]
    assert _pending(ledger) == []
    assert "jev1_a" not in rec._partial_armed


def test_partial_fill_below_stop_cancels_rest_and_closes(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.orders["jev1_a"].update(status="PARTIALLY_FILLED", executedQty="0.3", avgPrice="100")
    broker.fill_on_cancel["jev1_a"] = 0.3
    broker.trade(11, "BUY", 0.3, 100.0, datetime.now(timezone.utc))
    broker.long(0.3)
    broker.stop_answer = {"http_status": 400, "body": {"code": -2021, "msg": "Order would immediately trigger."}}

    _rec(ledger, broker).run_once()

    assert broker.cancelled == ["jev1_a"]
    assert [(f["action"], f["qty"]) for f in _fills(ledger)][0] == ("buy_long", 0.3)
    assert len(broker.submitted) == 1 and broker.submitted[0].reduce_only
    assert _pending(ledger) == []
