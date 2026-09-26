"""Reconcile pass: time budget, halt on 429/418/5xx/transport, short lock."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from jev_trader.execution import ExchangeHTTPError, order_lock
from jev_trader.ledger import Ledger
from jev_trader.reconcile import Reconciler

from tests.test_reconcile import SYM, FakeBroker, _fills, _record_entry


def _rec(ledger, broker, **kw) -> Reconciler:
    return Reconciler(broker, ledger, min_interval_sec=0, **kw)


class ThrottledBroker(FakeBroker):
    def __init__(self, status: int, retry_after: float | None = None) -> None:
        super().__init__()
        self.status = status
        self.retry_after = retry_after

    def query_order(self, symbol, cid=None, *, order_id=None):
        self.queries += 1
        body = {"code": -1003, "msg": "Too many requests"}
        if self.retry_after is not None:
            body["retry_after_sec"] = self.retry_after
        return self.status, body


def test_429_stops_pass_and_respects_retry_after(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = ThrottledBroker(429, retry_after=120)
    for i in range(3):
        _record_entry(ledger, broker, f"jev1_{i}", 10 + i)
    rec = _rec(ledger, broker)
    summary = rec.run_once()
    assert broker.queries == 1, "first 429 must stop the pass"
    assert summary["halted"]["http_status"] == 429
    assert summary["halted"]["wait_sec"] >= 120
    # Orders stay pending, nothing marked final.
    assert len(ledger.pending_orders(datetime.now(timezone.utc) - timedelta(hours=1))) == 3
    # maybe_run respects the backoff window.
    assert rec.maybe_run() is None
    assert broker.queries == 1


def test_backoff_doubles_and_resets_after_clean_pass(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = ThrottledBroker(418)
    _record_entry(ledger, broker, "jev1_a", 11)
    rec = _rec(ledger, broker)
    first = rec.run_once()["halted"]["wait_sec"]
    second = rec.run_once()["halted"]["wait_sec"]
    assert first == 10 and second == 20
    broker.status = 503
    assert rec.run_once()["halted"]["wait_sec"] == 40
    # Clean pass resets.
    broker.status = 200
    ThrottledBroker.query_order = FakeBroker.query_order  # type: ignore[method-assign]
    try:
        clean = rec.run_once()
        assert "halted" not in clean
        assert rec._backoff == 0.0
    finally:
        del ThrottledBroker.query_order


def test_transport_error_on_user_trades_halts_detect(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.orders["jev1_a"].update(status="FILLED", executedQty="1", avgPrice="100")

    def boom(symbol, **kw):
        raise ExchangeHTTPError("userTrades", 0, {"error": "timed out"})

    broker.user_trades = boom  # type: ignore[method-assign]
    summary = _rec(ledger, broker).run_once()
    assert summary["halted"]["what"] == "user_trades"
    # Not marked final: the fill is fetched again after the backoff.
    assert len(ledger.pending_orders(datetime.now(timezone.utc) - timedelta(hours=1))) == 1
    assert _fills(ledger) == []


def test_budget_stops_pending_loop(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")

    class SlowBroker(FakeBroker):
        def query_order(self, symbol, cid=None, *, order_id=None):
            time.sleep(0.06)
            return super().query_order(symbol, cid, order_id=order_id)

    broker = SlowBroker()
    for i in range(10):
        _record_entry(ledger, broker, f"jev1_{i}", 10 + i)
    summary = _rec(ledger, broker, budget_sec=0.1).run_once()
    assert summary.get("budget_exhausted") is True
    assert broker.queries < 10
    assert "halted" not in summary


def test_order_lock_not_held_during_exchange_query(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    seen: list[bool] = []

    class ProbeBroker(FakeBroker):
        def query_order(self, symbol, cid=None, *, order_id=None):
            got: list[bool] = []
            # Another thread (flatten) must be able to take the lock now.
            t = threading.Thread(target=lambda: got.append(order_lock.acquire(timeout=0.5)) or (got[-1] and order_lock.release()))
            t.start()
            t.join()
            seen.append(bool(got and got[0]))
            return super().query_order(symbol, cid, order_id=order_id)

    broker = ProbeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    _rec(ledger, broker).run_once()
    assert seen == [True]


def test_timeout_kw_only_for_brokers_that_accept_it(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    rec = _rec(ledger, FakeBroker())

    def with_timeout(symbol, cid=None, *, timeout=10.0):
        return timeout

    assert rec._timeout_kw(FakeBroker().query_order) == {}
    kw = rec._timeout_kw(with_timeout)
    assert 1.0 <= kw["timeout"] <= 10.0
