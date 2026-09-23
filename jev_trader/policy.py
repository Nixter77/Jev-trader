from __future__ import annotations

from jev_trader.models import (
    SIGNAL_STRENGTH_LEVELS,
    JevJudgment,
    PolicyDecision,
)

SHOULD_TRADE_NOW_MIN = 0.72
# Strict mode only. The profitable Jev desk closed at should ~0.34; follow_jev does too.
CLOSE_SHOULD_TRADE_MIN = 0.80
# Soft confirm when action_probabilities present. Ignored if empty, and ignored under follow_jev.
CLOSE_ACTION_PROB_MIN = 0.75
FALSE_BREAK_RISK_MAX = 0.35
SIGNAL_STRENGTH_MIN = "рабочий"
TREND_ALIGNED_MIN = 0.60
ENTRY_ACTIONS = frozenset({"buy_long"})
TRADE_ACTIONS = frozenset({"buy_long", "sell_short", "close"})


def signal_strength_rank(level: str) -> int:
    try:
        return SIGNAL_STRENGTH_LEVELS.index(level)
    except ValueError:
        return -1


def _action_prob(judgment: JevJudgment, action: str) -> float | None:
    probs = judgment.action_probabilities or {}
    if not probs:
        return None
    try:
        return float(probs.get(action, 0.0))
    except (TypeError, ValueError):
        return None


def _close_action_prob(judgment: JevJudgment) -> float | None:
    return _action_prob(judgment, "close")


def _hold(judgment: JevJudgment, reason: str) -> PolicyDecision:
    return PolicyDecision(action="hold", passed=False, skip_reason=reason, judgment=judgment)


def _allow(judgment: JevJudgment) -> PolicyDecision:
    return PolicyDecision(
        action=judgment.action, passed=True, skip_reason=None, judgment=judgment
    )


def _quality_skip(judgment: JevJudgment) -> str | None:
    """Shared false-break and strength bar. Close and entry both use it."""
    if judgment.false_break_risk > FALSE_BREAK_RISK_MAX:
        return "false_break"
    floor = signal_strength_rank(SIGNAL_STRENGTH_MIN)
    if signal_strength_rank(judgment.signal_strength) < floor:
        return "low_strength"
    return None


def apply_policy(
    judgment: JevJudgment,
    *,
    follow_jev: bool = False,
    min_should_trade: float = SHOULD_TRADE_NOW_MIN,
    close_should_trade_min: float = CLOSE_SHOULD_TRADE_MIN,
    close_action_prob_min: float = CLOSE_ACTION_PROB_MIN,
) -> PolicyDecision:
    """Confidence gates in code. Size/stop/leverage are never decided here.

    `follow_jev=True` is the desk Jev actually traded: buy and close both
    follow the model. Live Jev closes sat near should 0.34 (never 0.80) and
    that exit was the profitable part. A 0.80 close floor blocks it.
    `--strict-gates` still requires close should >= 0.80 and, when present,
    close probability >= 0.75. Risk flatten/stop bypasses policy via risk.py.
    """
    if judgment.action not in TRADE_ACTIONS:
        return _hold(judgment, "hold")

    # Same rule as the 20–21 Sep Jev book: the model's close is the exit.
    if follow_jev:
        return _allow(judgment)

    if judgment.action == "close":
        if judgment.should_trade_now < close_should_trade_min:
            return _hold(judgment, "close_should_trade_now")
        close_p = _close_action_prob(judgment)
        if close_p is not None and close_p < close_action_prob_min:
            return _hold(judgment, "close_action_prob")
        quality = _quality_skip(judgment)
        if quality:
            return _hold(judgment, quality)
        return _allow(judgment)

    if judgment.should_trade_now < min_should_trade:
        return _hold(judgment, "should_trade_now")
    quality = _quality_skip(judgment)
    if quality:
        return _hold(judgment, quality)
    if judgment.action in ENTRY_ACTIONS and judgment.trend_aligned < TREND_ALIGNED_MIN:
        return _hold(judgment, "trend_aligned")
    return _allow(judgment)
