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
    broker, calls = _broker((503, {"error": "down"}))
    intent = market_close_intent(symbol=SYM, qty=1.0, order_side="SELL", client_order_id="jf1_x", risk_event="kill_switch")
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
