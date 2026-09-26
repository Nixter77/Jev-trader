"""BNB (non-USDT) commission: logged, fill flagged, close PnL not marked exact."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jev_trader.ledger import Ledger
from jev_trader.reconcile import Reconciler

from tests.test_reconcile import SYM, FakeBroker, _fills, _record_entry


def _rec(ledger, broker):
    return Reconciler(broker, ledger, min_interval_sec=0)


def _bnb_trade(broker, order_id, side, qty, price, when, realized=0.0):
    broker.trade(order_id, side, qty, price, when, realized=realized)
    broker.trades[-1]["commissionAsset"] = "BNB"
    broker.trades[-1]["commission"] = "0.0001"


def test_bnb_entry_is_flagged_and_close_is_not_exact(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.orders["jev1_a"].update(status="FILLED", executedQty="1", avgPrice="100")
    now = datetime.now(timezone.utc)
    _bnb_trade(broker, 11, "BUY", 1.0, 100.0, now - timedelta(minutes=5))
    broker.long(1.0)
    rec = _rec(ledger, broker)
    rec.run_once()
    entry = _fills(ledger)[0]
    assert entry["fee_unaccounted"] == 1
    assert entry["commission_usdt"] == 0.0

    # Exchange stop closes with a USDT fee: still inexact, the entry fee was BNB.
    broker.flat()
    broker.trade(99, "SELL", 1.0, 95.0, now, commission=0.02, realized=-5.0)
    rec.run_once()
    close = [f for f in _fills(ledger) if f["action"] == "close"][0]
    # Its net includes the BNB entry fee as 0, so the close is flagged too.
    assert close["fee_unaccounted"] == 1
    assert close["pnl_source"] == "exchange_fee_unaccounted"


def test_usdt_only_round_trip_stays_exchange(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.orders["jev1_a"].update(status="FILLED", executedQty="1", avgPrice="100")
    now = datetime.now(timezone.utc)
    broker.trade(11, "BUY", 1.0, 100.0, now - timedelta(minutes=5))
    broker.long(1.0)
    rec = _rec(ledger, broker)
    rec.run_once()
    broker.flat()
    broker.trade(99, "SELL", 1.0, 95.0, now, commission=0.02, realized=-5.0)
    rec.run_once()
    fills = _fills(ledger)
    assert all(f["fee_unaccounted"] == 0 for f in fills)
    assert [f for f in fills if f["action"] == "close"][0]["pnl_source"] == "exchange"


def test_bnb_close_fee_flags_the_close(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.orders["jev1_a"].update(status="FILLED", executedQty="1", avgPrice="100")
    now = datetime.now(timezone.utc)
    broker.trade(11, "BUY", 1.0, 100.0, now - timedelta(minutes=5))
    broker.long(1.0)
    rec = _rec(ledger, broker)
    rec.run_once()
    broker.flat()
    _bnb_trade(broker, 99, "SELL", 1.0, 95.0, now, realized=-5.0)
    rec.run_once()
    close = [f for f in _fills(ledger) if f["action"] == "close"][0]
    assert close["fee_unaccounted"] == 1
    assert close["pnl_source"] == "exchange_fee_unaccounted"
