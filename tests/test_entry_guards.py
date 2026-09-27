from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from jev_trader.features import compute_features
from jev_trader.jev import judgment_from_dict
from jev_trader.ledger import Ledger
from jev_trader.models import AccountState, Position
from jev_trader.policy import apply_policy
from jev_trader.risk import (
    EntryGuardConfig,
    apply_risk,
    build_entry_guard_state,
    entry_guard_skip_reason,
    in_no_entry_window,
    load_entry_guard_config,
    loss_streak_from_closes,
    parse_no_entry_window,
)
from jev_trader.snapshot import build_fixture_snapshot


def _pass(action: str = "buy_long"):
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


def _account(**kwargs) -> AccountState:
    base = dict(
        equity_usdt=10_000.0,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=0,
        max_positions=3,
    )
    base.update(kwargs)
    return AccountState(**base)


def _cfg(**kwargs) -> EntryGuardConfig:
    base = dict(
        no_entry_window=(time(3, 0), time(9, 0)),
        no_entry_tz="Asia/Jerusalem",
        max_entries_per_hour=2,
        loss_streak_pause_n=3,
        loss_streak_pause_min=120.0,
    )
    base.update(kwargs)
    return EntryGuardConfig(**base)


def test_parse_window_default_and_off() -> None:
    assert parse_no_entry_window(None) == (time(3, 0), time(9, 0))
    assert parse_no_entry_window("03:00-09:00") == (time(3, 0), time(9, 0))
    assert parse_no_entry_window("off") is None
    assert parse_no_entry_window("") is None
    assert parse_no_entry_window("22:00-06:00") == (time(22, 0), time(6, 0))


def test_window_inside_outside_local() -> None:
    tz = "Asia/Jerusalem"
    # Pick a summer date so IL = UTC+3.
    inside = datetime(2026, 7, 15, 4, 30, tzinfo=ZoneInfo(tz))
    outside = datetime(2026, 7, 15, 10, 0, tzinfo=ZoneInfo(tz))
    assert in_no_entry_window(inside, time(3, 0), time(9, 0), tz) is True
    assert in_no_entry_window(outside, time(3, 0), time(9, 0), tz) is False


def test_window_wraps_midnight() -> None:
    tz = "UTC"
    late = datetime(2026, 1, 10, 23, 0, tzinfo=timezone.utc)
    early = datetime(2026, 1, 11, 5, 0, tzinfo=timezone.utc)
    midday = datetime(2026, 1, 11, 12, 0, tzinfo=timezone.utc)
    assert in_no_entry_window(late, time(22, 0), time(6, 0), tz) is True
    assert in_no_entry_window(early, time(22, 0), time(6, 0), tz) is True
    assert in_no_entry_window(midday, time(22, 0), time(6, 0), tz) is False


def test_window_dst_aware_jerusalem() -> None:
    """UTC midnight is 03:00 IDT (summer) but 02:00 IST (winter)."""
    summer_utc_midnight = datetime(2026, 7, 15, 0, 0, tzinfo=timezone.utc)
    winter_utc_midnight = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
    tz = "Asia/Jerusalem"
    assert in_no_entry_window(summer_utc_midnight, time(3, 0), time(9, 0), tz) is True
    assert in_no_entry_window(winter_utc_midnight, time(3, 0), time(9, 0), tz) is False


def test_load_config_defaults_and_disable(monkeypatch) -> None:
    monkeypatch.delenv("NO_ENTRY_WINDOW", raising=False)
    monkeypatch.delenv("NO_ENTRY_TZ", raising=False)
    monkeypatch.delenv("MAX_ENTRIES_PER_HOUR", raising=False)
    monkeypatch.delenv("LOSS_STREAK_PAUSE_N", raising=False)
    monkeypatch.delenv("LOSS_STREAK_PAUSE_MIN", raising=False)
    cfg = load_entry_guard_config()
    assert cfg.no_entry_window == (time(3, 0), time(9, 0))
    assert cfg.no_entry_tz == "Asia/Jerusalem"
    assert cfg.max_entries_per_hour == 2
    assert cfg.loss_streak_pause_n == 3
    assert cfg.loss_streak_pause_min == 120.0

    cfg2 = load_entry_guard_config(
        {
            "NO_ENTRY_WINDOW": "off",
            "MAX_ENTRIES_PER_HOUR": "0",
            "LOSS_STREAK_PAUSE_N": "0",
        }
    )
    assert cfg2.no_entry_window is None
    assert cfg2.max_entries_per_hour == 0
    assert cfg2.loss_streak_pause_n == 0


def test_loss_streak_counting_and_reset() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    closes = [
        (now, -5.0),
        (now - timedelta(minutes=10), -2.0),
        (now - timedelta(minutes=20), -1.0),
        (now - timedelta(minutes=30), 4.0),  # win resets older streak
        (now - timedelta(minutes=40), -9.0),
    ]
    streak, latest = loss_streak_from_closes(closes)
    assert streak == 3
    assert latest == now

    # Break-even resets
    streak2, _ = loss_streak_from_closes([(now, 0.0), (now - timedelta(minutes=1), -3.0)])
    assert streak2 == 0

    # Win resets
    streak3, _ = loss_streak_from_closes([(now, 1.0), (now - timedelta(minutes=1), -3.0)])
    assert streak3 == 0


def test_pause_until_active_and_expired() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    last_loss = now - timedelta(minutes=30)
    cfg = _cfg(loss_streak_pause_n=3, loss_streak_pause_min=120.0)
    active = build_entry_guard_state(
        config=cfg,
        now=now,
        entries_last_hour=0,
        loss_streak=3,
        last_loss_ts=last_loss,
    )
    assert active.pause_until == last_loss + timedelta(minutes=120)
    assert entry_guard_skip_reason(cfg, active, now=now) == "loss_streak_pause"

    expired = build_entry_guard_state(
        config=cfg,
        now=last_loss + timedelta(minutes=121),
        entries_last_hour=0,
        loss_streak=3,
        last_loss_ts=last_loss,
    )
    assert expired.pause_until is None
    assert entry_guard_skip_reason(cfg, expired, now=last_loss + timedelta(minutes=121)) is None


def test_no_entry_window_blocks_buy(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    now = datetime(2026, 7, 15, 4, 0, tzinfo=ZoneInfo("Asia/Jerusalem"))
    cfg = _cfg()
    state = build_entry_guard_state(
        config=cfg, now=now, entries_last_hour=0, loss_streak=0, last_loss_ts=None
    )
    intent = apply_risk(
        _pass(),
        features,
        market_snapshot,
        _account(),
        entry_guard_config=cfg,
        entry_guard_state=state,
        now=now,
    )
    assert intent.skip_reason == "no_entry_window"
    assert intent.qty == 0


def test_hourly_entry_cap_blocks_buy(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    now = datetime(2026, 7, 15, 12, 0, tzinfo=ZoneInfo("Asia/Jerusalem"))
    cfg = _cfg(max_entries_per_hour=2)
    state = build_entry_guard_state(
        config=cfg, now=now, entries_last_hour=2, loss_streak=0, last_loss_ts=None
    )
    intent = apply_risk(
        _pass(),
        features,
        market_snapshot,
        _account(),
        entry_guard_config=cfg,
        entry_guard_state=state,
        now=now,
    )
    assert intent.skip_reason == "hourly_entry_cap"


def test_daily_entry_cap_blocks_buy_and_spares_close(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    now = datetime(2026, 7, 15, 12, 0, tzinfo=ZoneInfo("Asia/Jerusalem"))
    cfg = _cfg(max_entries_per_hour=0, max_entries_per_day=3)
    state = build_entry_guard_state(
        config=cfg,
        now=now,
        entries_last_hour=0,
        loss_streak=0,
        last_loss_ts=None,
        entries_today=3,
    )
    blocked = apply_risk(
        _pass(),
        features,
        market_snapshot,
        _account(),
        entry_guard_config=cfg,
        entry_guard_state=state,
        now=now,
    )
    assert blocked.skip_reason == "daily_entry_cap"
    assert blocked.qty == 0

    long_snap = replace(
        market_snapshot,
        position=Position(
            side="LONG",
            size=1.0,
            cash_usdt=10_000.0,
            entry=float(market_snapshot.candles[-1].close),
        ),
    )
    closed = apply_risk(
        _pass("close"),
        compute_features(long_snap),
        long_snap,
        _account(open_positions=1),
        seconds_since_last_entry=10_000.0,
        entry_guard_config=cfg,
        entry_guard_state=state,
        now=now,
    )
    assert closed.action == "close"
    assert closed.entry_type == "LIMIT_POST_ONLY"
    assert closed.skip_reason is None


def test_hourly_cap_unlimited_allows(market_snapshot) -> None:
    features = compute_features(market_snapshot)
    now = datetime(2026, 7, 15, 12, 0, tzinfo=ZoneInfo("Asia/Jerusalem"))
    cfg = _cfg(max_entries_per_hour=0, no_entry_window=None, loss_streak_pause_n=0)
    state = build_entry_guard_state(
        config=cfg, now=now, entries_last_hour=99, loss_streak=0, last_loss_ts=None
    )
    intent = apply_risk(
        _pass(),
        features,
        market_snapshot,
        _account(),
        entry_guard_config=cfg,
        entry_guard_state=state,
        now=now,
    )
    assert intent.action == "buy_long"
    assert intent.skip_reason is None


def test_closes_still_allowed_during_all_guards(market_snapshot) -> None:
    long_snap = replace(
        market_snapshot,
        position=Position(
            side="LONG",
            size=1.0,
            cash_usdt=10_000.0,
            entry=float(market_snapshot.candles[-1].close),
        ),
    )
    features = compute_features(long_snap)
    now = datetime(2026, 7, 15, 4, 0, tzinfo=ZoneInfo("Asia/Jerusalem"))
    cfg = _cfg(max_entries_per_hour=1, loss_streak_pause_n=1, loss_streak_pause_min=120.0)
    state = build_entry_guard_state(
        config=cfg,
        now=now,
        entries_last_hour=5,
        loss_streak=5,
        last_loss_ts=now - timedelta(minutes=1),
    )
    # Discretionary close
    intent = apply_risk(
        _pass("close"),
        features,
        long_snap,
        _account(open_positions=1),
        seconds_since_last_entry=10_000.0,
        entry_guard_config=cfg,
        entry_guard_state=state,
        now=now,
    )
    assert intent.action == "close"
    assert intent.skip_reason is None
    assert intent.reduce_only is True
    assert intent.entry_type == "LIMIT_POST_ONLY"

    # Stop flatten
    stopped = replace(
        long_snap,
        position=replace(long_snap.position, stop_price=features.close + 1000.0),
    )
    stop_intent = apply_risk(
        _pass("hold"),
        compute_features(stopped),
        stopped,
        _account(open_positions=1),
        entry_guard_config=cfg,
        entry_guard_state=state,
        now=now,
    )
    assert stop_intent.action == "close"
    assert stop_intent.risk_event == "stop"
    assert stop_intent.entry_type == "MARKET"

    # Kill switch flatten
    kill_intent = apply_risk(
        _pass("buy_long"),
        features,
        long_snap,
        _account(open_positions=1, kill_switch=True),
        entry_guard_config=cfg,
        entry_guard_state=state,
        now=now,
    )
    assert kill_intent.action == "close"
    assert kill_intent.risk_event == "kill_switch"

    # Daily loss flatten
    daily_intent = apply_risk(
        _pass("buy_long"),
        features,
        long_snap,
        _account(open_positions=1, daily_pnl_pct=-0.03),
        entry_guard_config=cfg,
        entry_guard_state=state,
        now=now,
    )
    assert daily_intent.action == "close"
    assert daily_intent.risk_event == "daily_loss"


def test_ledger_counts_entries_and_streak(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    now = datetime.now(timezone.utc)
    with ledger._connect() as conn:
        # Two entries inside the hour, one older
        for i, age_min in enumerate([10, 30, 90]):
            conn.execute(
                """
                INSERT INTO fills (
                    ts, client_order_id, symbol, action, position_side,
                    qty, price, venue, realized_pnl_usdt, cash_usdt
                ) VALUES (?, ?, ?, 'buy_long', 'LONG', 1.0, 100.0, 'test', NULL, 10000.0)
                """,
                (
                    (now - timedelta(minutes=age_min)).isoformat(),
                    f"e{i}",
                    "BTCUSDT" if i < 2 else "ETHUSDT",
                ),
            )
        # Chronological insert (oldest first) so id DESC = newest first.
        # Newest three are losses; older win resets any prior streak.
        close_rows = [
            (now - timedelta(minutes=40), 5.0),   # older win
            (now - timedelta(minutes=30), -1.0),
            (now - timedelta(minutes=20), -2.0),
            (now - timedelta(minutes=10), -3.0),  # newest loss
        ]
        for i, (ts, pnl) in enumerate(close_rows):
            conn.execute(
                """
                INSERT INTO fills (
                    ts, client_order_id, symbol, action, position_side,
                    qty, price, venue, realized_pnl_usdt, cash_usdt
                ) VALUES (?, ?, 'BTCUSDT', 'close', 'LONG', 1.0, 100.0, 'test', ?, 10000.0)
                """,
                (ts.isoformat(), f"c{i}", pnl),
            )
        conn.commit()

    assert ledger.count_entries_since(now - timedelta(hours=1)) == 2
    yesterday = (now - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    with ledger._connect() as conn:
        conn.execute(
            """
            INSERT INTO fills (
                ts, client_order_id, symbol, action, position_side,
                qty, price, venue, realized_pnl_usdt, cash_usdt
            ) VALUES (?, 'old', 'BTCUSDT', 'buy_long', 'LONG', 1.0, 100.0, 'test', NULL, 10000.0)
            """,
            (yesterday.isoformat(),),
        )
        conn.commit()
    today_start = now.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    expected_today = sum(
        1 for age in (10, 30, 90) if now - timedelta(minutes=age) >= today_start
    )
    assert ledger.count_entries_today(now) == expected_today
    streak, latest = loss_streak_from_closes(ledger.recent_close_pnls(limit=16))
    assert streak == 3
    assert latest is not None


def test_guard_state_as_dict_shape() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    state = build_entry_guard_state(
        config=_cfg(no_entry_window=None),
        now=now,
        entries_last_hour=1,
        loss_streak=2,
        last_loss_ts=None,
    )
    payload = state.as_dict()
    assert payload == {
        "window_active": False,
        "entries_last_hour": 1,
        "loss_streak": 2,
        "pause_until": None,
        "max_entries_per_hour": 2,
        "loss_streak_pause_n": 3,
        "window_label": "выкл",
        "entries_today": 0,
        "max_entries_per_day": 3,
    }
