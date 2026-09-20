from __future__ import annotations

from jev_trader.models import (
    SIGNAL_STRENGTH_LEVELS,
    JevJudgment,
    PolicyDecision,
)

SHOULD_TRADE_NOW_MIN = 0.72
FOLLOW_JEV_ENTRY_MIN = 0.50
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


def apply_policy(
    judgment: JevJudgment,
    *,
    follow_jev: bool = False,
    min_should_trade: float = SHOULD_TRADE_NOW_MIN,
) -> PolicyDecision:
    """Confidence gates in code. Size/stop/leverage are never decided here.

    `follow_jev=True` still ignores hold and skips false-break/strength/trend
    gates, but buy_long still needs should_trade_now >= 0.50 (noise floor).
    Risk sizing stays in code.
    """
    if judgment.action not in TRADE_ACTIONS:
        return PolicyDecision(
            action="hold",
            passed=False,
            skip_reason="hold",
            judgment=judgment,
        )
    if follow_jev:
        if judgment.action in ENTRY_ACTIONS:
            floor = FOLLOW_JEV_ENTRY_MIN if min_should_trade == SHOULD_TRADE_NOW_MIN else min_should_trade
            if judgment.should_trade_now < floor:
                return PolicyDecision(
                    action="hold",
                    passed=False,
                    skip_reason="should_trade_now",
                    judgment=judgment,
                )
        return PolicyDecision(
            action=judgment.action,
            passed=True,
            skip_reason=None,
            judgment=judgment,
        )
    if judgment.should_trade_now < min_should_trade:
        return PolicyDecision(
            action="hold",
            passed=False,
            skip_reason="should_trade_now",
            judgment=judgment,
        )
    if judgment.false_break_risk > FALSE_BREAK_RISK_MAX:
        return PolicyDecision(
            action="hold",
            passed=False,
            skip_reason="false_break",
            judgment=judgment,
        )
    if signal_strength_rank(judgment.signal_strength) < signal_strength_rank(
        SIGNAL_STRENGTH_MIN
    ):
        return PolicyDecision(
            action="hold",
            passed=False,
            skip_reason="low_strength",
            judgment=judgment,
        )
    if judgment.action in ENTRY_ACTIONS and judgment.trend_aligned < TREND_ALIGNED_MIN:
        return PolicyDecision(
            action="hold",
            passed=False,
            skip_reason="trend_aligned",
            judgment=judgment,
        )
    return PolicyDecision(
        action=judgment.action,
        passed=True,
        skip_reason=None,
        judgment=judgment,
    )
