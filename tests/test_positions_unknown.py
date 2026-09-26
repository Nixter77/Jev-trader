"""Balance fallback wallet: equity is real, positions are unknown (not flat)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from jev_trader.execution import BinanceTestnetBroker, PaperBroker, wallet_positions_known
from jev_trader.jev import judgment_from_dict
from jev_trader.ledger import Ledger
from jev_trader.live import make_run_cycle
from jev_trader.models import TradeIntent, market_close_intent
from jev_trader.reconcile import Reconciler

SYM = "BTCUSDT"
BALANCE = [{"asset": "USDT", "balance": "1000", "availableBalance": "900"}]
UNKNOWN = {"equity_usdt": 1000.0, "available_usdt": 900.0, "open_positions": 0, "positions": [], "positions_known": False}


def _broker(account_answer: tuple[int, Any]) -> tuple[BinanceTestnetBroker, list[tuple[str, str]]]:
    broker = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    broker._time_synced = True
    broker.filters_for = lambda symbol: {}  # type: ignore[method-assign]
    calls: list[tuple[str, str]] = []

    def fake_request(method: str, path: str, params: dict | None = None, signed: bool = False, timeout: float = 10.0, **_kw: Any):
        calls.append((method, path))
        if path == "/fapi/v2/account":
            return account_answer
        if path == "/fapi/v2/balance":
            return 200, BALANCE
        return 200, []

    broker._request = fake_request  # type: ignore[method-assign]
    return broker, calls


def test_balance_fallback_wallet_marks_positions_unknown() -> None:
    broker, calls = _broker((503, {"error": "Service Unavailable"}))
    wallet = broker.fetch_wallet(ttl=0)
    assert ("GET", "/fapi/v2/balance") in calls
    assert wallet["equity_usdt"] == 1000.0
    assert wallet["positions_known"] is False
    assert not wallet_positions_known(wallet)


def test_account_wallet_marks_positions_known() -> None:
    broker, _ = _broker((200, {"totalMarginBalance": "1000", "availableBalance": "900", "positions": []}))
    wallet = broker.fetch_wallet(ttl=0)
    assert wallet["positions_known"] is True
    # Wallets without the flag (paper, fakes) are trusted.
    assert wallet_positions_known({"equity_usdt": 1.0, "positions": []})
    assert not wallet_positions_known(None)


def test_market_close_with_unknown_positions_is_rejected_not_flat() -> None:
    # A model close (no risk event) waits for a real positions list.
    broker, calls = _broker((503, {"error": "down"}))
    intent = market_close_intent(symbol=SYM, qty=1.0, order_side="SELL", client_order_id="jf1_x", risk_event=None)
    result = broker.submit(intent)
    assert result.status == "rejected"
    assert result.detail["error"] == "positions_unknown"
    assert ("POST", "/fapi/v1/order") not in calls


def test_flatten_with_unknown_positions_reports_it() -> None:
    broker, calls = _broker((503, {"error": "down"}))
    out = broker.flatten_all()
    assert out["error"] == "positions_unknown"
    assert out["closes"] == []


def _long_ledger(tmp_path) -> Ledger:
    ledger = Ledger(tmp_path / "l.sqlite")
    ledger.sync_exchange_positions({"equity_usdt": 1000.0, "positions": [{"symbol": SYM, "side": "LONG", "size": 1.0, "entry": 100.0}]})
    return ledger


def test_ledger_sync_skips_unknown_positions(tmp_path) -> None:
    ledger = _long_ledger(tmp_path)
    ledger.sync_exchange_positions(UNKNOWN)
    pos = ledger.load_position(SYM)
    assert pos.side == "LONG" and pos.size == 1.0
    assert ledger.count_open_positions() == 1


class BalanceOnlyBroker(PaperBroker):
    def fetch_wallet(self):
        return dict(UNKNOWN)


class CountingJudge:
    model = "fake"

    def __init__(self, answers: dict) -> None:
        self.answers = answers
        self.calls = 0

    def judge(self, _compact):
        self.calls += 1
        return judgment_from_dict(self.answers)


def test_cycle_with_unknown_positions_holds_and_keeps_ledger(tmp_path, market_snapshot, passing_answers) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    judge = CountingJudge(passing_answers)
    cycle = make_run_cycle(
        judgment=None,
        account_kwargs={},
        broker=BalanceOnlyBroker(),
        ledger=ledger,
        notifier=None,
        typesafe_api_key=None,
        jev_client=judge,
        wallet_box={},
    )
    result = cycle(market_snapshot)
    assert judge.calls == 0
    assert result.action == "hold"
    assert result.skip_reason == "wallet_unavailable"


def test_reconcile_does_not_burn_close_detection_on_unknown_positions(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
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
    trade_calls: list[Any] = []

    class B:
        venue = "binance_testnet"

        def fetch_wallet(self, *, ttl: float = 10.0):
            return dict(UNKNOWN)

        def user_trades(self, *a, **kw):
            trade_calls.append((a, kw))
            return []

        def query_order(self, *a, **kw):
            return 400, {"code": -2013}

    synced: list[Any] = []
    rec = Reconciler(B(), ledger, min_interval_sec=0, on_wallet=synced.append)
    for _ in range(5):
        rec.run_once()
    assert trade_calls == []
    assert synced == []
    assert rec._close_misses == {}


def _blind_broker(fill_qty: str) -> tuple[BinanceTestnetBroker, list[tuple[str, str, dict]]]:
    broker = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    broker._time_synced = True
    broker._fill_poll_sleep = 0
    broker.filters_for = lambda symbol: {}  # type: ignore[method-assign]
    calls: list[tuple[str, str, dict]] = []

    def fake_request(method: str, path: str, params: dict | None = None, signed: bool = False, timeout: float = 10.0, **_kw: Any):
        calls.append((method, path, dict(params or {})))
        if path == "/fapi/v2/account":
            return 503, {"error": "down"}
        if path == "/fapi/v2/balance":
            return 200, BALANCE
        if method == "POST" and path == "/fapi/v1/order":
            return 200, {"orderId": 91, "status": "FILLED", "executedQty": fill_qty, "avgPrice": "95"}
        return 200, []

    broker._request = fake_request  # type: ignore[method-assign]
    return broker, calls


def test_protective_close_with_unknown_positions_goes_out_reduce_only() -> None:
    # kill_switch / daily_loss / stop / flatten must not be refused because the
    # positions list is missing: a reduce-only SELL cannot open or flip.
    for event in ("kill_switch", "daily_loss", "stop", "flatten"):
        broker, calls = _blind_broker("1")
        intent = market_close_intent(symbol=SYM, qty=1.0, order_side="SELL", client_order_id=f"jc_{event}", risk_event=event)
        result = broker.submit(intent)
        assert result.status == "filled", event
        posts = [c for c in calls if c[0] == "POST" and c[1] == "/fapi/v1/order"]
        assert len(posts) == 1
        params = posts[0][2]
        assert params["side"] == "SELL" and params["reduceOnly"] == "true" and params["type"] == "MARKET"
        assert params["quantity"] == "1"
        # The exchange stop stays until we know the close flattened the book.
        assert not any(c[0] == "DELETE" for c in calls)


def test_blind_close_filled_short_of_last_known_qty_clears_the_stop() -> None:
    broker, calls = _blind_broker("0.4")
    intent = market_close_intent(symbol=SYM, qty=1.0, order_side="SELL", client_order_id="jc_k", risk_event="kill_switch")
    result = broker.submit(intent)
    assert result.status == "filled"
    post_i = next(i for i, c in enumerate(calls) if c[0] == "POST")
    assert any(c[0] == "DELETE" and c[1] == "/fapi/v1/allOpenOrders" for c in calls[post_i:])


def test_blind_cover_buy_is_still_refused() -> None:
    broker, calls = _blind_broker("1")
    intent = market_close_intent(symbol=SYM, qty=1.0, order_side="BUY", client_order_id="jc_s", risk_event="no_short")
    result = broker.submit(intent)
    assert result.status == "rejected" and result.detail["error"] == "positions_unknown"
    assert not any(c[0] == "POST" for c in calls)


def test_kill_switch_cycle_closes_last_known_long_when_positions_unknown(tmp_path, market_snapshot) -> None:
    ledger = _long_ledger(tmp_path)
    broker, calls = _blind_broker("1")
    judge = CountingJudge({})
    cycle = make_run_cycle(
        judgment=None,
        account_kwargs={"kill_switch": True},
        broker=broker,
        ledger=ledger,
        notifier=None,
        typesafe_api_key=None,
        jev_client=judge,
        wallet_box={},
    )
    snapshot = replace(market_snapshot, symbol=SYM, position=replace(market_snapshot.position, side="LONG", size=1.0, entry=100.0))
    result = cycle(snapshot)
    assert judge.calls == 0
    closes = [c for c in calls if c[0] == "POST" and c[1] == "/fapi/v1/order" and c[2].get("type") == "MARKET"]
    assert len(closes) == 1 and closes[0][2]["reduceOnly"] == "true" and closes[0][2]["side"] == "SELL"
    assert result.action == "close" and result.risk_event == "kill_switch"


def test_flatten_with_unknown_positions_closes_ledger_longs_and_keeps_stops(tmp_path) -> None:
    from jev_trader.flatten import flatten_open_positions

    ledger = _long_ledger(tmp_path)
    broker, calls = _blind_broker("1")
    out = flatten_open_positions(broker, ledger)
    assert out["error"] == "positions_unknown" and out["fallback"] == "ledger_positions"
    posts = [c for c in calls if c[0] == "POST" and c[1] == "/fapi/v1/order"]
    assert len(posts) == 1
    assert posts[0][2]["side"] == "SELL" and posts[0][2]["reduceOnly"] == "true" and posts[0][2]["quantity"] == "1"
    # Stops were never cancelled wholesale.
    assert not any(c[0] == "DELETE" and c[1] == "/fapi/v1/allOpenOrders" for c in calls)
    closes = [f for f in ledger.book()["fills"] if f["action"] == "close"]
    assert len(closes) == 1
