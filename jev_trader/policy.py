from __future__ import annotations

from jev_trader.models import (
    SIGNAL_STRENGTH_LEVELS,
    JevJudgment,
    PolicyDecision,
)

SHOULD_TRADE_NOW_MIN = 0.72
# Close needs a higher bar than entry (Laya typed-decisions was closing at ~0.35–0.50 under follow_jev).
CLOSE_SHOULD_TRADE_MIN = 0.80
# Soft confirm when action_probabilities present (Laya). Ignored if empty (classic Jev).
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

    `follow_jev=True` skips probability gates for **entries**. Live Laya
    buy_long sits near should 0.43 and trend 0.34, so a 0.50/0.60 floor
    admits nothing. **Close is always gated** (asymmetry): Laya was dumping
    losers while should_trade_now sat ~0.4. Risk flatten/stop still bypasses
    this via risk.py, not policy.
    """
    if judgment.action not in TRADE_ACTIONS:
        return _hold(judgment, "hold")

    # Close stays gated under follow_jev. Entries may skip the probability bar.
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

    if follow_jev:
        return _allow(judgment)
    if judgment.should_trade_now < min_should_trade:
        return _hold(judgment, "should_trade_now")
    quality = _quality_skip(judgment)
    if quality:
        return _hold(judgment, quality)
    if judgment.action in ENTRY_ACTIONS and judgment.trend_aligned < TREND_ALIGNED_MIN:
        return _hold(judgment, "trend_aligned")
    return _allow(judgment)
