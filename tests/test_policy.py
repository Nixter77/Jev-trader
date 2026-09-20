from __future__ import annotations

import pytest

from jev_trader.jev import judgment_from_dict
from jev_trader.policy import apply_policy


def _judgment(**overrides):
    base = {
        "action": "buy_long",
        "trend_aligned": 0.81,
        "false_break_risk": 0.22,
        "signal_strength": "рабочий",
        "should_trade_now": 0.84,
    }
    base.update(overrides)
    return judgment_from_dict(base)


def test_passing_fixture_yields_trade_intent() -> None:
    decision = apply_policy(_judgment())
    assert decision.passed is True
    assert decision.action == "buy_long"
    assert decision.skip_reason is None


def test_hold_action_skips() -> None:
    decision = apply_policy(_judgment(action="hold"))
    assert decision.passed is False
    assert decision.action == "hold"
    assert decision.skip_reason == "hold"


def test_follow_jev_skips_probability_gates() -> None:
    decision = apply_policy(
        _judgment(should_trade_now=0.36, false_break_risk=0.9, signal_strength="слабый"),
        follow_jev=True,
    )
    assert decision.passed is True
    assert decision.action == "buy_long"


def test_follow_jev_still_holds_on_hold_action() -> None:
    decision = apply_policy(_judgment(action="hold"), follow_jev=True)
    assert decision.passed is False
    assert decision.skip_reason == "hold"


def test_should_trade_now_gate() -> None:
    decision = apply_policy(_judgment(should_trade_now=0.71))
    assert decision.passed is False
    assert decision.skip_reason == "should_trade_now"


def test_false_break_gate() -> None:
    decision = apply_policy(_judgment(false_break_risk=0.36))
    assert decision.passed is False
    assert decision.skip_reason == "false_break"


def test_low_strength_gate() -> None:
    decision = apply_policy(_judgment(signal_strength="слабый"))
    assert decision.passed is False
    assert decision.skip_reason == "low_strength"


def test_trend_aligned_gate_on_entry() -> None:
    decision = apply_policy(_judgment(trend_aligned=0.59))
    assert decision.passed is False
    assert decision.skip_reason == "trend_aligned"


def test_sell_short_does_not_use_entry_trend_gate() -> None:
    decision = apply_policy(_judgment(action="sell_short", trend_aligned=0.10))
    assert decision.passed is True
    assert decision.action == "sell_short"


def test_close_relaxes_trend_aligned() -> None:
    decision = apply_policy(_judgment(action="close", trend_aligned=0.10))
    assert decision.passed is True
    assert decision.action == "close"
    assert decision.skip_reason is None


@pytest.mark.parametrize("level", ["рабочий", "сильный"])
def test_working_or_strong_passes_strength(level: str) -> None:
    assert apply_policy(_judgment(signal_strength=level)).passed is True
