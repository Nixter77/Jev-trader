from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from jev_trader.cycle import recorded_outcome
from jev_trader.ledger import Ledger
from jev_trader.models import CycleResult, ExecutionResult, TradeIntent
from jev_trader.reconcile import Reconciler
from jev_trader.risk import loss_streak_from_closes

SYM = "BTCUSDT"


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class FakeBroker:
    venue = "binance_testnet"

    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {}
        self.trades: list[dict[str, Any]] = []
        self.funding = 0.0
        self.wallet: dict[str, Any] = {"equity_usdt": 1000.0, "positions": []}
        self.cancelled: list[str] = []
        self.fill_on_cancel: dict[str, float] = {}
        self.stops: list[tuple[str, float]] = []
        self.stop_answer: dict[str, Any] = {"http_status": 200, "body": {"orderId": 9}}
        self.submitted: list[TradeIntent] = []
        self.queries = 0
        self.trade_calls = 0

    def query_order(self, symbol, cid=None, *, order_id=None):
        self.queries += 1
        body = self.orders.get(cid)
        if body is None:
            return 400, {"code": -2013, "msg": "Order does not exist."}
        return 200, dict(body)

    def cancel_order(self, symbol, cid):
        self.cancelled.append(cid)
        body = self.orders[cid]
        filled = self.fill_on_cancel.get(cid, 0.0)
        body["status"] = "CANCELED"
        if filled:
            body["executedQty"] = str(filled)
            body["avgPrice"] = body["price"]
        return {"http_status": 200, "body": body}

    def user_trades(self, symbol, *, order_id=None, start_ms=None, end_ms=None, limit=1000):
        self.trade_calls += 1
        rows = [t for t in self.trades if t["symbol"] == symbol]
        if order_id is not None:
            return [t for t in rows if str(t["orderId"]) == str(order_id)]
        return [t for t in rows if (start_ms is None or t["time"] >= start_ms) and (end_ms is None or t["time"] <= end_ms)]

    def funding_income(self, symbol, start_ms, end_ms):
        return self.funding

    def invalidate_wallet(self):
        pass

    def fetch_wallet(self, *, ttl: float = 10.0):
        return self.wallet

    def ensure_stop_market(self, symbol, *, stop_price, order_side="SELL"):
        self.stops.append((symbol, stop_price))
        return self.stop_answer

    def submit(self, intent: TradeIntent) -> ExecutionResult:
        self.submitted.append(intent)
        return ExecutionResult(
            status="filled",
            venue=self.venue,
            client_order_id=intent.client_order_id,
            reduce_only=True,
            detail={"http_status": 200, "body": {"orderId": 777, "status": "FILLED", "executedQty": str(intent.qty), "avgPrice": "94"}},
        )

    def long(self, qty: float, entry: float = 100.0) -> None:
        self.wallet = {"equity_usdt": 1000.0, "positions": [{"symbol": SYM, "side": "LONG", "size": qty, "entry": entry}]}

    def flat(self) -> None:
        self.wallet = {"equity_usdt": 1000.0, "positions": []}

    def trade(self, order_id, side, qty, price, when, *, commission=0.01, realized=0.0) -> None:
        self.trades.append(
            {
                "symbol": SYM,
                "orderId": order_id,
                "side": side,
                "qty": str(qty),
                "price": str(price),
                "commission": str(commission),
                "commissionAsset": "USDT",
                "realizedPnl": str(realized),
                "time": _ms(when),
            }
        )


def _record_entry(ledger: Ledger, broker: FakeBroker, cid: str, order_id: int, *, stop: float = 95.0, qty: float = 1.0) -> None:
    body = {"orderId": order_id, "status": "NEW", "executedQty": "0", "price": "100", "avgPrice": "0"}
    broker.orders[cid] = dict(body)
    intent = TradeIntent(
        action="buy_long",
        qty=qty,
        stop_price=stop,
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
            execution=ExecutionResult(status="working", venue="binance_testnet", client_order_id=cid, reduce_only=False, detail={"http_status": 200, "body": body}),
            judgment=None,
            state_text="",
            state={"symbol": SYM},
        )
    )


def _rec(ledger: Ledger, broker: FakeBroker, *, ahead: float = 0.0) -> Reconciler:
    return Reconciler(broker, ledger, min_interval_sec=0, now_fn=lambda: datetime.now(timezone.utc) + timedelta(seconds=ahead))


def _fills(ledger: Ledger) -> list[dict[str, Any]]:
    with ledger._connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM fills ORDER BY ts, id")]


def test_resting_entry_counts_against_hourly_cap(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    assert ledger.count_entries_since(hour_ago) == 1
    # Young resting order: not cancelled, still pending.
    summary = _rec(ledger, broker).run_once()
    assert broker.cancelled == []
    assert summary["entries"] == []
    assert ledger.count_entries_since(hour_ago) == 1


def test_maker_fill_is_recorded_and_stop_armed_at_once(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    now = datetime.now(timezone.utc)
    broker.orders["jev1_a"].update(status="FILLED", executedQty="1", avgPrice="100")
    broker.trade(11, "BUY", 0.4, 99.0, now, commission=0.02)
    broker.trade(11, "BUY", 0.6, 100.0, now, commission=0.03)
    broker.long(1.0)

    summary = _rec(ledger, broker).run_once()

    assert [e["cid"] for e in summary["entries"]] == ["jev1_a"]
    fills = _fills(ledger)
    assert len(fills) == 1
    fill = fills[0]
    assert fill["action"] == "buy_long" and fill["source"] == "exchange"
    assert fill["price"] == pytest.approx(99.6)
    assert fill["commission_usdt"] == pytest.approx(0.05)
    assert fill["exchange_order_id"] == "11"
    assert broker.stops == [(SYM, 95.0)]
    assert ledger.load_position(SYM).stop_price == 95.0
    assert ledger.pending_orders(now - timedelta(hours=1)) == []
    # Counted once (fill), not twice (fill + pending).
    assert ledger.count_entries_since(now - timedelta(hours=1)) == 1
    assert ledger.seconds_since_last_entry(SYM) is not None

    again = _rec(ledger, broker).run_once()
    assert again["entries"] == [] and len(_fills(ledger)) == 1
    assert broker.stops == [(SYM, 95.0)]


def test_stale_entry_is_cancelled_and_partial_fill_recorded(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    _record_entry(ledger, broker, "jev1_b", 12)
    broker.fill_on_cancel["jev1_b"] = 0.3
    later = datetime.now(timezone.utc) + timedelta(minutes=6)
    broker.trade(12, "BUY", 0.3, 100.0, later, commission=0.01)
    broker.long(0.3)

    summary = _rec(ledger, broker, ahead=360).run_once()

    assert sorted(broker.cancelled) == ["jev1_a", "jev1_b"]
    assert [e["cid"] for e in summary["entries"]] == ["jev1_b"]
    fills = _fills(ledger)
    assert [(f["client_order_id"], f["qty"]) for f in fills] == [("jev1_b", 0.3)]
    with ledger._connect() as conn:
        finals = dict(conn.execute("SELECT client_order_id, final_status FROM orders").fetchall())
    assert finals == {"jev1_a": "canceled", "jev1_b": "canceled"}
    assert broker.stops == [(SYM, 95.0)]
    # The cancelled one no longer counts, the filled one does.
    assert ledger.count_entries_since(datetime.now(timezone.utc) - timedelta(hours=1)) == 1


def test_order_unknown_to_exchange_is_settled_not_found(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    del broker.orders["jev1_a"]
    _rec(ledger, broker).run_once()
    assert ledger.pending_orders(datetime.now(timezone.utc) - timedelta(hours=1)) == []
    assert ledger.count_entries_since(datetime.now(timezone.utc) - timedelta(hours=1)) == 0


def test_exchange_stop_close_recorded_with_exchange_pnl(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    t0 = datetime.now(timezone.utc) - timedelta(minutes=30)
    broker.orders["jev1_a"].update(status="FILLED", executedQty="1", avgPrice="100")
    broker.trade(11, "BUY", 1.0, 100.0, t0, commission=0.04)
    broker.long(1.0)
    rec = _rec(ledger, broker)
    rec.run_once()

    # Exchange STOP_MARKET fires: wallet flat, a SELL the bot never sent.
    broker.flat()
    broker.trade(55, "SELL", 1.0, 95.0, t0 + timedelta(minutes=10), commission=0.05, realized=-5.0)
    broker.funding = -0.1
    summary = rec.run_once()

    assert [c["cid"] for c in summary["closes"]] == ["x55"]
    close = [f for f in _fills(ledger) if f["action"] == "close"][0]
    # gross -5 - close fee 0.05 - entry fee 0.04 + funding -0.1
    assert close["realized_pnl_usdt"] == pytest.approx(-5.19)
    assert close["gross_pnl_usdt"] == pytest.approx(-5.0)
    assert close["funding_usdt"] == pytest.approx(-0.1)
    assert close["pnl_source"] == "exchange"
    streak, _ = loss_streak_from_closes(ledger.recent_close_pnls())
    assert streak == 1

    third = rec.run_once()
    assert third["closes"] == []
    assert len([f for f in _fills(ledger) if f["action"] == "close"]) == 1


def test_flat_wallet_without_sell_trades_gives_up(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    ledger.record_exchange_fill(
        ts=datetime.now(timezone.utc).isoformat(),
        client_order_id="jev1_a",
        exchange_order_id=11,
        symbol=SYM,
        action="buy_long",
        qty=1.0,
        price=100.0,
        venue="binance_testnet",
        commission_usdt=0.01,
    )
    rec = _rec(ledger, broker)
    for _ in range(5):
        rec.run_once()
    assert broker.trade_calls == 3


def test_bot_close_pnl_replaced_by_exchange_numbers(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    now = datetime.now(timezone.utc)

    def bot_fill(action: str, cid: str, oid: int, price: float) -> None:
        intent = TradeIntent(
            action=action,
            qty=1.0,
            stop_price=95.0 if action == "buy_long" else None,
            stop_distance=None,
            entry_type="MARKET",
            reduce_only=action == "close",
            client_order_id=cid,
            symbol=SYM,
            risk_pct=0.0,
            order_side="BUY" if action == "buy_long" else "SELL",
        )
        ledger.record(
            CycleResult(
                action=action,
                skip_reason=None,
                intent=intent,
                execution=ExecutionResult(
                    status="filled",
                    venue="binance_testnet",
                    client_order_id=cid,
                    reduce_only=action == "close",
                    detail={"http_status": 200, "body": {"orderId": oid, "status": "FILLED", "executedQty": "1", "avgPrice": str(price)}},
                ),
                judgment=None,
                state_text="",
                state={"symbol": SYM},
            )
        )

    bot_fill("buy_long", "jev1_in", 21, 100.0)
    bot_fill("close", "jev1_out", 22, 101.0)
    local = [f for f in _fills(ledger) if f["action"] == "close"][0]
    assert local["realized_pnl_usdt"] == pytest.approx(1.0)
    assert local["pnl_source"] == "local"
    broker.trade(21, "BUY", 1.0, 100.0, now, commission=0.6)
    broker.trade(22, "SELL", 1.0, 101.0, now, commission=0.6, realized=1.0)

    summary = _rec(ledger, broker).run_once()

    assert summary["enriched"] == 2
    close = [f for f in _fills(ledger) if f["action"] == "close"][0]
    assert close["realized_pnl_usdt"] == pytest.approx(-0.2)
    assert close["commission_usdt"] == pytest.approx(0.6)
    assert close["pnl_source"] == "exchange"
    # A fee-negative "win" now counts as a loss for the streak.
    streak, _ = loss_streak_from_closes(ledger.recent_close_pnls())
    assert streak == 1
    with ledger._connect() as conn:
        realized_total = conn.execute("SELECT realized_pnl_usdt FROM positions WHERE symbol=?", (SYM,)).fetchone()[0]
    assert realized_total == pytest.approx(-0.2)
    assert _rec(ledger, broker).run_once()["enriched"] == 0


def test_stop_that_would_trigger_closes_market(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.orders["jev1_a"].update(status="FILLED", executedQty="1", avgPrice="100")
    broker.trade(11, "BUY", 1.0, 100.0, datetime.now(timezone.utc))
    broker.long(1.0)
    broker.stop_answer = {"http_status": 400, "body": {"code": -2021, "msg": "Order would immediately trigger."}}

    summary = _rec(ledger, broker).run_once()

    assert summary["stops"][0]["fail_closed"] == "filled"
    assert len(broker.submitted) == 1
    close = broker.submitted[0]
    assert close.action == "close" and close.reduce_only and close.qty == 1.0
    closes = [f for f in _fills(ledger) if f["action"] == "close"]
    assert len(closes) == 1 and closes[0]["price"] == pytest.approx(94.0)


def test_query_errors_leave_order_pending(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_a", 11)
    broker.query_order = lambda *a, **k: (503, {"error": "busy"})  # type: ignore[assignment]
    summary = _rec(ledger, broker, ahead=600).run_once()
    assert summary["errors"]
    assert len(ledger.pending_orders(datetime.now(timezone.utc) - timedelta(hours=1))) == 1


def test_submit_unknown_is_a_hold() -> None:
    execution = ExecutionResult(status="submit_unknown", venue="binance_testnet", client_order_id="c", reduce_only=False)
    intent = TradeIntent(
        action="buy_long",
        qty=1.0,
        stop_price=95.0,
        stop_distance=5.0,
        entry_type="LIMIT_POST_ONLY",
        reduce_only=False,
        client_order_id="c",
        symbol=SYM,
        risk_pct=0.005,
    )
    policy = SimpleNamespace(passed=True, skip_reason=None, action="buy_long")
    assert recorded_outcome(policy, intent, execution) == ("hold", "submit_unknown")


def test_live_runner_reconciles_before_each_symbol() -> None:
    from jev_trader.live import LiveRunner

    class Stub:
        calls = 0

        def maybe_run(self):
            Stub.calls += 1
            if Stub.calls == 1:
                raise RuntimeError("boom")  # must not stop the pass
            return {"entries": [{"cid": "a"}], "closes": [], "cancelled": [], "stops": [], "enriched": 0, "errors": []}

    runner = LiveRunner(["BTCUSDT", "ETHUSDT"], run_cycle=lambda *a, **k: None, reconciler=Stub())
    polled: list[str] = []
    runner.poll_live_symbol = lambda symbol: polled.append(symbol)  # type: ignore[method-assign]
    runner.poll_once(emit=False)
    assert polled == ["BTCUSDT", "ETHUSDT"]
    assert Stub.calls == 2
    assert runner.last_reconcile is not None and runner.last_reconcile["entries"] == 1


def test_old_backfilled_entry_does_not_move_current_stop(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    _record_entry(ledger, broker, "jev1_old", 11, stop=90.0)
    t0 = datetime.now(timezone.utc) - timedelta(hours=2)
    broker.orders["jev1_old"].update(status="FILLED", executedQty="1", avgPrice="100")
    broker.trade(11, "BUY", 1.0, 100.0, t0)
    # A newer entry already filled and was recorded by the bot's own cycle.
    ledger.record_exchange_fill(
        ts=(t0 + timedelta(hours=1)).isoformat(),
        client_order_id="jev1_new",
        exchange_order_id=12,
        symbol=SYM,
        action="buy_long",
        qty=1.0,
        price=100.0,
        venue="binance_testnet",
        commission_usdt=0.01,
    )
    broker.long(1.0)
    summary = _rec(ledger, broker).run_once()
    assert [e["cid"] for e in summary["entries"]] == ["jev1_old"]
    assert broker.stops == []


def test_bot_fill_with_known_order_id_is_not_recorded_twice(tmp_path) -> None:
    # The reconciler settled the exchange close as x777; a flatten running in
    # another process (own order_lock) then records the same orderId.
    ledger = Ledger(tmp_path / "l.sqlite")
    broker = FakeBroker()
    t0 = datetime.now(timezone.utc) - timedelta(minutes=30)
    ledger.record_exchange_fill(
        ts=t0.isoformat(),
        client_order_id="jev1_in",
        exchange_order_id=1,
        symbol=SYM,
        action="buy_long",
        qty=1.0,
        price=100.0,
        venue="binance_testnet",
        commission_usdt=0.04,
    )
    broker.flat()
    broker.trade(777, "SELL", 1.0, 94.0, t0 + timedelta(minutes=5), commission=0.05, realized=-6.0)
    _rec(ledger, broker).run_once()
    intent = TradeIntent(
        action="close",
        qty=1.0,
        stop_price=None,
        stop_distance=None,
        entry_type="MARKET",
        reduce_only=True,
        client_order_id="jfBTCUSDT1",
        symbol=SYM,
        risk_pct=0.0,
        order_side="SELL",
        limit_price=94.0,
    )
    ledger.record(
        CycleResult(
            action="close",
            skip_reason=None,
            intent=intent,
            execution=ExecutionResult(
                status="filled",
                venue="binance_testnet",
                client_order_id="jfBTCUSDT1",
                reduce_only=True,
                detail={"http_status": 200, "body": {"orderId": 777, "status": "FILLED", "executedQty": "1", "avgPrice": "94"}},
            ),
            judgment=None,
            state_text="flatten",
            state={"symbol": SYM, "price": {"close": 94.0}},
            risk_event="flatten",
        )
    )
    closes = [f for f in _fills(ledger) if f["action"] == "close"]
    assert [c["client_order_id"] for c in closes] == ["x777"]
    streak, _ = loss_streak_from_closes(ledger.recent_close_pnls())
    assert streak == 1
