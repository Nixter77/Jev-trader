from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time
from zoneinfo import ZoneInfo

from jev_trader.cycle import run_once
from jev_trader.desk import skip_reason_counts
from jev_trader.execution import PaperBroker
from jev_trader.features import compute_features
from jev_trader.jev import judgment_from_dict
from jev_trader.live import make_run_cycle
from jev_trader.models import AccountState, Position
from jev_trader.policy import apply_policy
from jev_trader.risk import EntryGuardConfig, apply_risk

NOON = datetime(2026, 7, 15, 12, 0, tzinfo=ZoneInfo("Asia/Jerusalem"))


class CountingJudge:
    model = "fake"

    def __init__(self, answers: dict) -> None:
        self.answers = answers
        self.calls = 0

    def judge(self, _compact):
        self.calls += 1
        return judgment_from_dict(self.answers)


class NoWalletBroker(PaperBroker):
    def fetch_wallet(self):
        raise RuntimeError("wallet HTTP 503")


def _cfg() -> EntryGuardConfig:
    return EntryGuardConfig(
        no_entry_window=(time(3, 0), time(9, 0)),
        no_entry_tz="Asia/Jerusalem",
        max_entries_per_hour=0,
        loss_streak_pause_n=0,
        loss_streak_pause_min=0.0,
    )


def _account(**kw) -> AccountState:
    base = {"equity_usdt": 10_000.0, "daily_pnl_pct": 0.0, "kill_switch": False, "open_positions": 0}
    base.update(kw)
    return AccountState(**base)


def _buy_policy():
    return apply_policy(
        judgment_from_dict(
            {
                "action": "buy_long",
                "trend_aligned": 0.9,
                "false_break_risk": 0.1,
                "signal_strength": "сильный",
                "should_trade_now": 0.95,
            }
        )
    )


def test_flat_book_without_wallet_skips_model(market_snapshot, passing_answers) -> None:
    judge = CountingJudge(passing_answers)
    result = run_once(
        market_snapshot,
        jev_client=judge,
        account=_account(wallet_ok=False),
        broker=PaperBroker(),
        entry_guard_config=_cfg(),
        now=NOON,
    )
    assert judge.calls == 0
    assert result.action == "hold"
    assert result.skip_reason == "wallet_unavailable"
    assert result.model_skipped is True
    assert skip_reason_counts([{"skip_reason": result.skip_reason}])["wallet_unavailable"] == 1


def test_open_position_without_wallet_still_calls_model(market_snapshot, passing_answers) -> None:
    close = float(market_snapshot.candles[-1].close)
    snap = replace(
        market_snapshot,
        position=Position(side="LONG", size=1.0, cash_usdt=10_000.0, entry=close),
    )
    judge = CountingJudge(passing_answers)
    result = run_once(
        snap,
        jev_client=judge,
        account=_account(wallet_ok=False, open_positions=1),
        broker=PaperBroker(),
        entry_guard_config=_cfg(),
        now=NOON,
    )
    assert judge.calls == 1
    assert result.model_skipped is False


def test_risk_blocks_buy_without_wallet(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    intent = apply_risk(_buy_policy(), features, market_snapshot, _account(wallet_ok=False))
    assert intent.action == "hold"
    assert intent.skip_reason == "wallet_unavailable"
    assert intent.qty == 0


def test_zero_available_is_insufficient_margin(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    intent = apply_risk(_buy_policy(), features, market_snapshot, _account(available_usdt=0.0))
    assert intent.action == "hold"
    assert intent.skip_reason == "insufficient_margin"
    negative = apply_risk(_buy_policy(), features, market_snapshot, _account(available_usdt=-5.0))
    assert negative.skip_reason == "insufficient_margin"
    ok = apply_risk(_buy_policy(), features, market_snapshot, _account(available_usdt=1_000.0))
    assert ok.action == "buy_long"
    assert ok.qty * features.close <= 1_000.0 * 3.0 + 1e-6


def test_live_cycle_marks_wallet_unavailable_and_logs(market_snapshot, passing_answers, capsys) -> None:
    judge = CountingJudge(passing_answers)
    box: dict = {}
    cycle = make_run_cycle(
        judgment=None,
        account_kwargs={},
        broker=NoWalletBroker(),
        ledger=None,
        notifier=None,
        typesafe_api_key=None,
        jev_client=judge,
        wallet_box=box,
    )
    result = cycle(market_snapshot)
    assert judge.calls == 0
    assert result.skip_reason == "wallet_unavailable"
    assert "wallet HTTP 503" in (box.get("wallet_error") or "")
    assert '"wallet_error"' in capsys.readouterr().err


def test_paper_broker_without_wallet_endpoint_still_trades(market_snapshot, passing_answers) -> None:
    judge = CountingJudge(passing_answers)
    cycle = make_run_cycle(
        judgment=None,
        account_kwargs={},
        broker=PaperBroker(),
        ledger=None,
        notifier=None,
        typesafe_api_key=None,
        jev_client=judge,
    )
    result = cycle(market_snapshot)
    assert judge.calls == 1
    assert result.skip_reason != "wallet_unavailable"
