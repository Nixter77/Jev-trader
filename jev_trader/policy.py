from __future__ import annotations

from jev_trader.models import (
    SIGNAL_STRENGTH_LEVELS,
    JevJudgment,
    PolicyDecision,
)

SHOULD_TRADE_NOW_MIN = 0.72
# Strict mode and Laya closes. Trial 2026-09-23: 0.80 left Laya unable to exit
# (0 fills). Exchange income for 20–27 Sep 2026 is the scoreboard, not the desk.
CLOSE_SHOULD_TRADE_MIN = 0.45
# Soft confirm when action_probabilities present (typical for Laya).
CLOSE_ACTION_PROB_MIN = 0.55
FALSE_BREAK_RISK_MAX = 0.35
# Laya closes often sit just above entry false-break; slightly looser on gated close only.
CLOSE_FALSE_BREAK_RISK_MAX = 0.45
SIGNAL_STRENGTH_MIN = "рабочий"
TREND_ALIGNED_MIN = 0.60
ENTRY_ACTIONS = frozenset({"buy_long"})
TRADE_ACTIONS = frozenset({"buy_long", "sell_short", "close"})


def signal_strength_rank(level: str) -> int:
    try:
        return SIGNAL_STRENGTH_LEVELS.index(level)
    except ValueError:
        return -1


def is_laya_model(judgment: JevJudgment) -> bool:
    """True for local Laya backends (`laya:…`). Jev cloud ids stay False."""
    return str(judgment.model or "").strip().lower().startswith("laya")


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


def _gated_close(
    judgment: JevJudgment,
    *,
    close_should_trade_min: float,
    close_action_prob_min: float,
    close_false_break_max: float = CLOSE_FALSE_BREAK_RISK_MAX,
) -> PolicyDecision:
    if judgment.should_trade_now < close_should_trade_min:
        return _hold(judgment, "close_should_trade_now")
    close_p = _close_action_prob(judgment)
    if close_p is not None and close_p < close_action_prob_min:
        return _hold(judgment, "close_action_prob")
    if judgment.false_break_risk > close_false_break_max:
        return _hold(judgment, "false_break")
    floor = signal_strength_rank(SIGNAL_STRENGTH_MIN)
    if signal_strength_rank(judgment.signal_strength) < floor:
        return _hold(judgment, "low_strength")
    return _allow(judgment)


def apply_policy(
    judgment: JevJudgment,
    *,
    follow_jev: bool = False,
    min_should_trade: float = SHOULD_TRADE_NOW_MIN,
    close_should_trade_min: float = CLOSE_SHOULD_TRADE_MIN,
    close_action_prob_min: float = CLOSE_ACTION_PROB_MIN,
) -> PolicyDecision:
    """Confidence gates in code. Size/stop/leverage are never decided here.

    Exchange income 20–27 Sep 2026 (testnet): about −717 USDT, of which
    commission was −499. The desk ledger had called Jev's soft closes
    profitable; that book omitted fees. `follow_jev` is the explicit flag
    for that week:
      - Jev: buy and close follow the model.
      - Laya: entries follow the model; close stays on the trial floors
        (0.45 / 0.55 / false_break 0.45).
    `run` defaults to follow_jev=False, which gates both backends the same way.
    Risk flatten/stop bypasses policy via risk.py.
    """
    if judgment.action not in TRADE_ACTIONS:
        return _hold(judgment, "hold")

    if judgment.action == "close":
        # Backend-aware asymmetry under follow_jev.
        if follow_jev and not is_laya_model(judgment):
            return _allow(judgment)
        return _gated_close(
            judgment,
            close_should_trade_min=close_should_trade_min,
            close_action_prob_min=close_action_prob_min,
        )

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
