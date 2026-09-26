"""Item 6: close-detect misses reset on a timer; funding fetch retried once."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jev_trader.ledger import Ledger
from jev_trader.reconcile import Reconciler

from tests.test_reconcile import SYM, FakeBroker, _fills, _record_entry


def _open_entry(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.orders["jev1_a"].update(status="FILLED", executedQty="1", avgPrice="100")
    now = datetime.now(timezone.utc)
    broker.trade(11, "BUY", 1.0, 100.0, now - timedelta(minutes=10))
    return ledger, broker


def test_close_miss_counter_resets_after_timer(tmp_path) -> None:
    ledger, broker = _open_entry(tmp_path)
    rec = Reconciler(broker, ledger, min_interval_sec=0, close_miss_reset_sec=0.0)
    broker.long(1.0)
    rec.run_once()  # records the entry
    broker.flat()
    for _ in range(3):
        rec.run_once()
    # With a 0 s timer the gave-up entry is retried on the next pass.
    calls = broker.trade_calls
    rec.run_once()
    assert broker.trade_calls > calls


def test_close_miss_counter_stays_without_timer(tmp_path) -> None:
    ledger, broker = _open_entry(tmp_path)
    rec = Reconciler(broker, ledger, min_interval_sec=0)
    broker.long(1.0)
    rec.run_once()
    broker.flat()
    for _ in range(3):
        rec.run_once()
    calls = broker.trade_calls
    rec.run_once()
    assert broker.trade_calls == calls
    assert rec._close_misses == {"jev1_a": 3}


def test_funding_fetch_is_retried_once(tmp_path) -> None:
    ledger, broker = _open_entry(tmp_path)
    answers = [None, 0.25]
    seen: list[int] = []

    def funding(symbol, start_ms, end_ms):
        seen.append(1)
        return answers.pop(0)

    broker.funding_income = funding  # type: ignore[method-assign]
    rec = Reconciler(broker, ledger, min_interval_sec=0)
    broker.long(1.0)
    rec.run_once()
    broker.flat()
    broker.trade(99, "SELL", 1.0, 95.0, datetime.now(timezone.utc), commission=0.02, realized=-5.0)
    rec.run_once()
    close = [f for f in _fills(ledger) if f["action"] == "close"][0]
    assert len(seen) == 2
    assert close["funding_usdt"] == 0.25
