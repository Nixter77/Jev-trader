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
    broker.long(1.0)

    def boom(symbol, **kw):
        raise ExchangeHTTPError("userTrades", 0, {"error": "timed out"})

    broker.user_trades = boom  # type: ignore[method-assign]
    summary = _rec(ledger, broker).run_once()
    assert summary["halted"]["what"] == "user_trades"
    # A FILLED entry is a live long: the fill is written from the order body
    # (fee later, via enrichment) and its stop arms despite the halt.
    assert ledger.pending_orders(datetime.now(timezone.utc) - timedelta(hours=1)) == []
    fills = _fills(ledger)
    assert len(fills) == 1 and fills[0]["qty"] == 1.0 and fills[0]["price"] == 100.0
    assert fills[0]["commission_usdt"] is None
    assert summary["entries"][0]["fee_pending"] is True
    assert summary["stops"] and summary["stops"][0]["armed"] is True
    assert broker.stops == [(SYM, 95.0)]


def test_transport_error_on_user_trades_for_close_stays_pending(tmp_path) -> None:
    from jev_trader.models import CycleResult, ExecutionResult, market_close_intent

    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    intent = market_close_intent(symbol=SYM, qty=1.0, order_side="SELL", client_order_id="jc1_a", risk_event=None)
    ledger.record(
        CycleResult(
            action="close",
            skip_reason=None,
            intent=intent,
            execution=ExecutionResult(status="submit_unknown", venue="binance_testnet", client_order_id="jc1_a", reduce_only=True, detail={}),
            judgment=None,
            state_text="",
            state={"symbol": SYM},
        )
    )
    broker.orders["jc1_a"] = {"orderId": 12, "status": "FILLED", "executedQty": "1", "avgPrice": "101", "symbol": SYM}

    def boom(symbol, **kw):
        raise ExchangeHTTPError("userTrades", 0, {"error": "timed out"})

    broker.user_trades = boom  # type: ignore[method-assign]
    summary = _rec(ledger, broker).run_once()
    assert summary["halted"]["what"] == "user_trades"
    # A close needs its trades for PnL: stays pending until after the backoff.
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


def test_fill_recorded_before_a_halt_still_gets_its_stop(tmp_path) -> None:
    # Newest row fills and is marked final; the next (older) row hits a 503.
    # The final fill is never re-queried, so its stop must be armed now.
    ledger = Ledger(tmp_path / "l.sqlite")

    class HalfDown(FakeBroker):
        def query_order(self, symbol, cid=None, *, order_id=None):
            if cid == "jev1_old":
                self.queries += 1
                return 503, {"error": "Service Unavailable"}
            return super().query_order(symbol, cid, order_id=order_id)

    broker = HalfDown()
    _record_entry(ledger, broker, "jev1_old", 10)
    time.sleep(0.01)
    _record_entry(ledger, broker, "jev1_new", 11)
    broker.orders["jev1_new"].update(status="FILLED", executedQty="1", avgPrice="100")
    broker.trade(11, "BUY", 1.0, 100.0, datetime.now(timezone.utc))
    broker.long(1.0)
    summary = _rec(ledger, broker).run_once()
    assert summary["halted"]["http_status"] == 503
    assert [f["client_order_id"] for f in _fills(ledger)] == ["jev1_new"]
    assert broker.stops == [(SYM, 95.0)]


def test_no_stop_arming_during_ip_ban(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")

    class Banned(FakeBroker):
        def query_order(self, symbol, cid=None, *, order_id=None):
            if cid == "jev1_old":
                return 418, {"code": -1003, "msg": "banned"}
            return super().query_order(symbol, cid, order_id=order_id)

    broker = Banned()
    _record_entry(ledger, broker, "jev1_old", 10)
    time.sleep(0.01)
    _record_entry(ledger, broker, "jev1_new", 11)
    broker.orders["jev1_new"].update(status="FILLED", executedQty="1", avgPrice="100")
    broker.trade(11, "BUY", 1.0, 100.0, datetime.now(timezone.utc))
    broker.long(1.0)
    summary = _rec(ledger, broker).run_once()
    assert summary["halted"]["http_status"] == 418
    assert broker.stops == []


def test_detect_keeps_its_reserve_when_pending_is_slow(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")

    class SlowBroker(FakeBroker):
        def query_order(self, symbol, cid=None, *, order_id=None):
            time.sleep(0.05)
            return super().query_order(symbol, cid, order_id=order_id)

    broker = SlowBroker()
    for i in range(10):
        _record_entry(ledger, broker, f"jev1_{i}", 10 + i)
    rec = _rec(ledger, broker, budget_sec=0.4, detect_reserve_sec=0.2)
    ran: list[float] = []
    rec._detect_exchange_closes = lambda now, summary: ran.append(rec._time_left())  # type: ignore[method-assign]
    summary = rec.run_once()
    assert summary.get("budget_exhausted") is True
    assert broker.queries < 10
    assert len(ran) == 1 and ran[0] > 0.1  # detect still ran, with most of its reserve


def test_detect_reserve_is_capped_at_half_the_budget(tmp_path) -> None:
    rec = _rec(Ledger(tmp_path / "l.sqlite"), FakeBroker(), budget_sec=2.0, detect_reserve_sec=5.0)
    assert rec.detect_reserve_sec == 1.0
