"""Flat book + active entry guard ⇒ no model call (no Jev/Laya tokens)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from jev_trader.cycle import MODEL_SKIPPED, decision_payload, run_once
from jev_trader.desk import skip_reason_counts
from jev_trader.execution import PaperBroker
from jev_trader.jev import judgment_from_dict
from jev_trader.models import DAILY_LOSS_LIMIT_PCT, AccountState, Position
from jev_trader.risk import EntryGuardConfig

IL = ZoneInfo("Asia/Jerusalem")
NIGHT = datetime(2026, 7, 15, 4, 0, tzinfo=IL)  # inside 03:00–09:00
NOON = datetime(2026, 7, 15, 12, 0, tzinfo=IL)


class CountingJudge:
    model = "fake"

    def __init__(self, answers: dict) -> None:
        self.answers = answers
        self.calls = 0

    def judge(self, _compact):
        self.calls += 1
        return judgment_from_dict(self.answers)


class FakeLedger:
    def __init__(self, *, entries_last_hour: int = 0, closes=None) -> None:
        self.entries_last_hour = entries_last_hour
        self.closes = list(closes or [])
        self.recorded = []

    def seconds_since_last_close(self, _symbol):
        return None

    def seconds_since_last_entry(self, _symbol):
        return None

    def count_entries_since(self, _since):
        return self.entries_last_hour

    def recent_close_pnls(self, *, limit: int = 64):
        return self.closes[:limit]

    def set_stop(self, _symbol, _price):
        return None

    def record(self, result):
        self.recorded.append(result)


def _cfg(**kw) -> EntryGuardConfig:
    base = {
        "no_entry_window": (time(3, 0), time(9, 0)),
        "no_entry_tz": "Asia/Jerusalem",
        "max_entries_per_hour": 2,
        "loss_streak_pause_n": 3,
        "loss_streak_pause_min": 120.0,
    }
    base.update(kw)
    return EntryGuardConfig(**base)


def _account(**kw) -> AccountState:
    base = {"equity_usdt": 10_000.0, "daily_pnl_pct": 0.0, "kill_switch": False, "open_positions": 0}
    base.update(kw)
    return AccountState(**base)


def _long(snapshot):
    close = float(snapshot.candles[-1].close)
    return replace(
        snapshot,
        position=Position(side="LONG", size=1.0, cash_usdt=10_000.0, entry=close),
    )


def _losing_closes(now: datetime, n: int = 3):
    return [(now - timedelta(minutes=5 + i), -1.0) for i in range(n)]


GUARDS = ("daily_loss", "no_entry_window", "hourly_entry_cap", "loss_streak_pause")


def _case(guard: str) -> dict:
    """Fresh account / ledger / clock that activates exactly one guard."""
    if guard == "daily_loss":
        return dict(
            account=_account(daily_pnl_pct=-(DAILY_LOSS_LIMIT_PCT + 0.005)),
            ledger=FakeLedger(),
            now=NOON,
        )
    if guard == "no_entry_window":
        return dict(account=_account(), ledger=FakeLedger(), now=NIGHT)
    if guard == "hourly_entry_cap":
        return dict(account=_account(), ledger=FakeLedger(entries_last_hour=2), now=NOON)
    if guard == "loss_streak_pause":
        return dict(account=_account(), ledger=FakeLedger(closes=_losing_closes(NOON)), now=NOON)
    raise AssertionError(guard)


@pytest.mark.parametrize("guard", GUARDS)
def test_flat_with_active_guard_skips_model(market_snapshot, passing_answers, guard) -> None:
    case = _case(guard)
    judge = CountingJudge(passing_answers)
    ledger = case["ledger"]
    result = run_once(
        market_snapshot,
        jev_client=judge,
        account=case["account"],
        broker=PaperBroker(),
        ledger=ledger,
        entry_guard_config=_cfg(),
        now=case["now"],
    )
    assert market_snapshot.position.side == "FLAT"
    assert judge.calls == 0
    assert result.judge_ms is None
    assert result.model_skipped is True
    assert result.action == "hold"
    assert result.skip_reason == guard
    assert result.risk_event == guard
    assert result.intent.qty == 0
    assert result.execution.status == "skipped"
    assert result.judgment.model == MODEL_SKIPPED
    assert result.judgment.action == "hold"
    # Still recorded so monitor / status show it.
    assert ledger.recorded == [result]
    payload = decision_payload(result)
    assert payload["model_skipped"] is True
    assert "judge_ms" not in payload
    assert payload["skip_reason"] == guard
    assert skip_reason_counts([{"skip_reason": payload["skip_reason"]}])[guard] == 1


def test_priority_daily_loss_first_then_window(market_snapshot, passing_answers) -> None:
    judge = CountingJudge(passing_answers)
    ledger = FakeLedger(entries_last_hour=5, closes=_losing_closes(NIGHT))
    result = run_once(
        market_snapshot,
        jev_client=judge,
        account=_account(daily_pnl_pct=-0.05),
        broker=PaperBroker(),
        ledger=ledger,
        entry_guard_config=_cfg(),
        now=NIGHT,
    )
    assert judge.calls == 0
    assert result.skip_reason == "daily_loss"

    result = run_once(
        market_snapshot,
        jev_client=judge,
        account=_account(),
        broker=PaperBroker(),
        ledger=FakeLedger(entries_last_hour=5, closes=_losing_closes(NIGHT)),
        entry_guard_config=_cfg(),
        now=NIGHT,
    )
    assert judge.calls == 0
    assert result.skip_reason == "no_entry_window"


@pytest.mark.parametrize("guard", [g for g in GUARDS if g != "daily_loss"])
def test_open_position_with_guard_still_calls_model(market_snapshot, passing_answers, guard) -> None:
    case = _case(guard)
    judge = CountingJudge({**passing_answers, "action": "hold"})
    result = run_once(
        _long(market_snapshot),
        jev_client=judge,
        account=replace(case["account"], open_positions=1),
        broker=PaperBroker(),
        ledger=case["ledger"],
        entry_guard_config=_cfg(),
        now=case["now"],
    )
    assert judge.calls == 1
    assert result.judge_ms is not None
    assert result.model_skipped is False
    assert result.judgment.model != MODEL_SKIPPED
    # daily_loss with an open long closes without the model (see
    # test_daily_loss_open_position_closes_even_if_model_is_down).


def test_no_guards_calls_model(market_snapshot, passing_answers) -> None:
    judge = CountingJudge(passing_answers)
    result = run_once(
        market_snapshot,
        jev_client=judge,
        account=_account(),
        broker=PaperBroker(),
        ledger=FakeLedger(),
        entry_guard_config=_cfg(),
        now=NOON,
    )
    assert judge.calls == 1
    assert result.judge_ms is not None
    assert result.model_skipped is False
    assert "model_skipped" not in decision_payload(result)
    assert result.skip_reason != "no_entry_window"


def test_kill_switch_flat_skips_model(market_snapshot, passing_answers) -> None:
    judge = CountingJudge(passing_answers)
    ledger = FakeLedger()
    result = run_once(
        market_snapshot,
        jev_client=judge,
        account=_account(kill_switch=True, daily_pnl_pct=-0.05),
        broker=PaperBroker(),
        ledger=ledger,
        entry_guard_config=_cfg(),
        now=NIGHT,
    )
    assert judge.calls == 0
    assert result.action == "hold"
    assert result.skip_reason == "kill_switch"
    assert result.risk_event == "kill_switch"
    assert result.model_skipped is True
    assert ledger.recorded == [result]


def test_kill_switch_open_position_closes_without_model(market_snapshot, passing_answers) -> None:
    judge = CountingJudge({**passing_answers, "action": "hold"})
    result = run_once(
        _long(market_snapshot),
        jev_client=judge,
        account=_account(kill_switch=True, open_positions=1),
        broker=PaperBroker(),
        ledger=FakeLedger(),
        entry_guard_config=_cfg(),
        now=NOON,
    )
    assert judge.calls == 0
    assert result.model_skipped is True
    assert result.risk_event == "kill_switch"
    assert result.intent.action == "close"
    assert result.intent.entry_type == "MARKET"
    assert result.intent.reduce_only is True


class BrokenJudge:
    model = "broken"

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, _compact):
        self.calls += 1
        raise RuntimeError("model down")


class FillingBroker:
    venue = "binance_testnet"

    def __init__(self) -> None:
        self.sent = []

    def submit(self, intent):
        from jev_trader.models import ExecutionResult

        self.sent.append(intent)
        return ExecutionResult(
            status="filled",
            venue=self.venue,
            client_order_id=intent.client_order_id,
            reduce_only=intent.reduce_only,
            detail={"http_status": 200, "body": {"orderId": 1, "status": "FILLED", "executedQty": str(intent.qty), "avgPrice": "100"}, "filled_qty": intent.qty, "fill_price": 100.0},
        )


def test_daily_loss_open_position_closes_even_if_model_is_down(market_snapshot) -> None:
    judge = BrokenJudge()
    broker = FillingBroker()
    result = run_once(
        _long(market_snapshot),
        jev_client=judge,
        account=_account(daily_pnl_pct=-(DAILY_LOSS_LIMIT_PCT + 0.01), open_positions=1),
        broker=broker,
        ledger=FakeLedger(),
        entry_guard_config=_cfg(),
        now=NOON,
    )
    assert judge.calls == 0
    assert [i.action for i in broker.sent] == ["close"]
    assert broker.sent[0].entry_type == "MARKET" and broker.sent[0].reduce_only
    assert result.action == "close"
    assert result.skip_reason is None
    assert result.risk_event == "daily_loss"
    assert result.model_skipped is True


def test_precomputed_judgment_path_unchanged(market_snapshot, passing_answers) -> None:
    result = run_once(
        market_snapshot,
        judgment=judgment_from_dict(passing_answers),
        account=_account(),
        broker=PaperBroker(),
        ledger=FakeLedger(),
        entry_guard_config=_cfg(),
        now=NIGHT,
    )
    assert result.model_skipped is False
    assert result.judgment.model != MODEL_SKIPPED
    assert result.skip_reason == "no_entry_window"
