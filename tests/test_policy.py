from __future__ import annotations

import pytest

from jev_trader.jev import judgment_from_dict
from jev_trader.policy import apply_policy, is_laya_model


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


def test_follow_jev_buys_at_live_laya_scores() -> None:
    """Live typed-decisions buy_long is ~0.43 should and ~0.34 trend. That is an entry."""
    decision = apply_policy(
        _judgment(
            should_trade_now=0.43,
            trend_aligned=0.34,
            model="laya:typed-decisions",
            action_probabilities={"buy_long": 0.56, "hold": 0.44},
        ),
        follow_jev=True,
    )
    assert decision.passed is True
    assert decision.action == "buy_long"
    assert decision.skip_reason is None


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
    decision = apply_policy(
        _judgment(
            action="close",
            trend_aligned=0.10,
            should_trade_now=0.85,
            action_probabilities={"close": 0.82, "hold": 0.18},
        )
    )
    assert decision.passed is True
    assert decision.action == "close"
    assert decision.skip_reason is None


@pytest.mark.parametrize("level", ["рабочий", "сильный"])
def test_working_or_strong_passes_strength(level: str) -> None:
    assert apply_policy(_judgment(signal_strength=level)).passed is True


def test_is_laya_model_detects_prefix() -> None:
    assert is_laya_model(_judgment(model="laya:typed-decisions")) is True
    assert is_laya_model(_judgment(model="jev-1.13.0")) is False


def test_follow_jev_closes_at_live_jev_scores() -> None:
    """Jev's profitable exits were close at should ~0.34, false-break well above 0.35."""
    decision = apply_policy(
        _judgment(
            action="close",
            should_trade_now=0.34,
            trend_aligned=0.55,
            false_break_risk=0.62,
            signal_strength="рабочий",
            model="jev-1.13.0",
            action_probabilities={"close": 0.55, "hold": 0.45},
        ),
        follow_jev=True,
    )
    assert decision.passed is True
    assert decision.action == "close"
    assert decision.skip_reason is None


def test_follow_jev_blocks_soft_laya_close() -> None:
    """Laya soft closes bled the book; gate even under follow_jev."""
    decision = apply_policy(
        _judgment(
            action="close",
            should_trade_now=0.44,
            trend_aligned=0.55,
            false_break_risk=0.22,
            signal_strength="рабочий",
            model="laya:typed-decisions",
            action_probabilities={"close": 0.65, "hold": 0.35},
        ),
        follow_jev=True,
    )
    assert decision.passed is False
    assert decision.skip_reason == "close_should_trade_now"


def test_follow_jev_allows_strong_laya_close() -> None:
    decision = apply_policy(
        _judgment(
            action="close",
            should_trade_now=0.85,
            trend_aligned=0.10,
            model="laya:typed-decisions",
            action_probabilities={"close": 0.82, "hold": 0.18},
        ),
        follow_jev=True,
    )
    assert decision.passed is True
    assert decision.action == "close"


def test_close_requires_higher_should_than_entry() -> None:
    # Entry bar 0.72 would pass buy; close needs CLOSE_SHOULD_TRADE_MIN (0.80).
    decision = apply_policy(
        _judgment(action="close", should_trade_now=0.75, trend_aligned=0.10)
    )
    assert decision.passed is False
    assert decision.skip_reason == "close_should_trade_now"


def test_close_action_prob_gate_when_probs_present() -> None:
    decision = apply_policy(
        _judgment(
            action="close",
            should_trade_now=0.90,
            action_probabilities={"close": 0.60, "hold": 0.40},
        ),
    )
    assert decision.passed is False
    assert decision.skip_reason == "close_action_prob"


def test_close_without_probs_skips_prob_gate_for_jev_follow() -> None:
    decision = apply_policy(
        _judgment(
            action="close",
            should_trade_now=0.34,
            model="jev-1.13.0",
            trend_aligned=0.10,
        ),
        follow_jev=True,
    )
    assert decision.passed is True
    assert decision.action == "close"
