from __future__ import annotations

from dataclasses import replace

import pytest

from jev_trader.features import compute_features
from jev_trader.jev import judgment_from_dict
from jev_trader.models import AccountState
from jev_trader.policy import apply_policy
from jev_trader.risk import apply_risk, post_only_limit_price
from jev_trader.snapshot import build_fixture_snapshot


def _pass_policy(action: str = "buy_long"):
    return apply_policy(
        judgment_from_dict(
            {
                "action": action,
                "trend_aligned": 0.9,
                "false_break_risk": 0.1,
                "signal_strength": "сильный",
                "should_trade_now": 0.95,
            }
        )
    )


def test_size_from_risk_pct_and_atr_stop_not_jev_probability(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    account = AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=0,
        risk_pct=0.005,
        atr_stop_mult=1.5,
    )
    high_p = apply_risk(_pass_policy(), features, market_snapshot, account)
    low_p_policy = apply_policy(
        judgment_from_dict(
            {
                "action": "buy_long",
                "trend_aligned": 0.61,
                "false_break_risk": 0.34,
                "signal_strength": "рабочий",
                "should_trade_now": 0.72,
            }
        )
    )
    low_p = apply_risk(low_p_policy, features, market_snapshot, account)

    assert high_p.skip_reason is None
    assert high_p.qty > 0
    assert high_p.qty == low_p.qty
    stop_distance = 1.5 * features.atr
    assert high_p.stop_distance == pytest.approx(stop_distance)
    assert high_p.qty * high_p.stop_distance == pytest.approx(10_000.0 * 0.005)
    assert high_p.qty != 0.95 * 10_000.0
    assert high_p.entry_type == "LIMIT_POST_ONLY"
    assert high_p.reduce_only is False
    assert high_p.limit_price == market_snapshot.book.bids[0].price
    assert high_p.limit_price == post_only_limit_price("BUY", features.close, market_snapshot)


def test_kill_switch_blocks_new_orders(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    account = AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=True,
        open_positions=0,
    )
    intent = apply_risk(_pass_policy(), features, market_snapshot, account)
    assert intent.skip_reason == "kill_switch"
    assert intent.qty == 0


def test_kill_switch_flattens_open_position() -> None:
    snapshot = build_fixture_snapshot(side="LONG", size=0.25)
    features = compute_features(snapshot)
    account = AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=True,
        open_positions=1,
    )
    intent = apply_risk(_pass_policy(), features, snapshot, account)
    assert intent.action == "close"
    assert intent.reduce_only is True
    assert intent.entry_type == "MARKET"
    assert intent.qty == 0.25
    assert intent.risk_event == "kill_switch"
    assert intent.skip_reason is None


def test_daily_loss_flatten_requested() -> None:
    snapshot = build_fixture_snapshot(side="SHORT", size=0.1)
    features = compute_features(snapshot)
    account = AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=-0.03,
        kill_switch=False,
        open_positions=1,
        daily_loss_limit_pct=0.025,
    )
    intent = apply_risk(_pass_policy("hold"), features, snapshot, account)
    assert intent.action == "close"
    assert intent.reduce_only is True
    assert intent.risk_event == "daily_loss"
    assert intent.order_side == "BUY"


def test_reduce_only_set_on_close() -> None:
    snapshot = build_fixture_snapshot(side="LONG", size=0.07)
    features = compute_features(snapshot)
    account = AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=1,
    )
    intent = apply_risk(_pass_policy("close"), features, snapshot, account)
    assert intent.action == "close"
    assert intent.reduce_only is True
    assert intent.qty == 0.07
    assert intent.order_side == "SELL"
    assert intent.limit_price == snapshot.book.asks[0].price
    assert intent.entry_type == "LIMIT_POST_ONLY"


def test_sell_short_while_flat_does_not_open(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    account = AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=0,
    )
    intent = apply_risk(_pass_policy("sell_short"), features, market_snapshot, account)
    assert intent.qty == 0
    assert intent.skip_reason == "no_short"
    assert intent.reduce_only is False


def test_sell_short_while_long_closes_with_sell() -> None:
    snapshot = build_fixture_snapshot(side="LONG", size=0.4)
    features = compute_features(snapshot)
    account = AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=1,
    )
    intent = apply_risk(_pass_policy("sell_short"), features, snapshot, account)
    assert intent.action == "close"
    assert intent.order_side == "SELL"
    assert intent.reduce_only is True
    assert intent.qty == 0.4
    assert intent.limit_price == snapshot.book.asks[0].price


def test_leftover_short_is_flattened_even_on_hold() -> None:
    snapshot = build_fixture_snapshot(side="SHORT", size=0.2)
    features = compute_features(snapshot)
    account = AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=1,
    )
    intent = apply_risk(_pass_policy("hold"), features, snapshot, account)
    assert intent.action == "close"
    assert intent.order_side == "BUY"
    assert intent.reduce_only is True
    assert intent.risk_event == "no_short"
    assert intent.entry_type == "MARKET"


def test_buy_long_skips_when_already_long() -> None:
    snapshot = build_fixture_snapshot(side="LONG", size=0.3)
    features = compute_features(snapshot)
    account = AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=1,
    )
    intent = apply_risk(_pass_policy("buy_long"), features, snapshot, account)
    assert intent.skip_reason == "already_long"
    assert intent.qty == 0


def test_limit_price_falls_back_to_last_close_without_book(market_snapshot) -> None:
    snapshot = replace(market_snapshot, book=None)
    features = compute_features(snapshot)
    account = AccountState(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=0,
    )
    intent = apply_risk(_pass_policy(), features, snapshot, account)
    assert intent.limit_price == features.close
