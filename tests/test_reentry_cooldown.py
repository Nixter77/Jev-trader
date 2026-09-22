from __future__ import annotations

from jev_trader.features import compute_features
from jev_trader.models import AccountState
from jev_trader.policy import apply_policy
from jev_trader.jev import judgment_from_dict
from jev_trader.risk import REENTRY_COOLDOWN_SEC, apply_risk


def _pass_buy_policy():
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


def _account() -> AccountState:
    return AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=0,
        max_positions=3,
    )


def test_reentry_cooldown_blocks_buy_after_recent_close(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    intent = apply_risk(
        _pass_buy_policy(),
        features,
        market_snapshot,
        _account(),
        seconds_since_last_close=60.0,
        reentry_cooldown_sec=1800.0,
    )
    assert intent.action == "hold"
    assert intent.skip_reason == "reentry_cooldown"
    assert intent.qty == 0


def test_reentry_allowed_after_cooldown(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    intent = apply_risk(
        _pass_buy_policy(),
        features,
        market_snapshot,
        _account(),
        seconds_since_last_close=REENTRY_COOLDOWN_SEC + 1.0,
        reentry_cooldown_sec=REENTRY_COOLDOWN_SEC,
    )
    assert intent.action == "buy_long"
    assert intent.skip_reason is None
    assert intent.qty > 0


def test_reentry_allowed_when_never_closed(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    intent = apply_risk(
        _pass_buy_policy(),
        features,
        market_snapshot,
        _account(),
        seconds_since_last_close=None,
    )
    assert intent.action == "buy_long"
    assert intent.skip_reason is None
