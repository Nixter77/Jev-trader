from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from jev_trader.execution import client_order_id
from jev_trader.models import (
    DAILY_LOSS_LIMIT_PCT,
    AccountState,
    Features,
    MarketSnapshot,
    PolicyDecision,
    TradeIntent,
    market_close_intent,
)

ATR_STOP_MULT_MIN = 1.2
ATR_STOP_MULT_MAX = 1.8

# After a close, wait before re-entering the same symbol (anti-chop).
REENTRY_COOLDOWN_SEC = 1800.0
# Block discretionary model closes until the entry has aged (3×5m bars).
# Exchange stop / kill / daily_loss still fire immediately above.
MIN_HOLD_SEC = 900.0

# Entry guards (entries only — closes / stops / kill / daily_loss / no_short flatten
# are evaluated earlier in apply_risk and never consult these).
DEFAULT_NO_ENTRY_WINDOW = "03:00-09:00"
DEFAULT_NO_ENTRY_TZ = "Asia/Jerusalem"
DEFAULT_MAX_ENTRIES_PER_HOUR = 2
DEFAULT_LOSS_STREAK_PAUSE_N = 3
DEFAULT_LOSS_STREAK_PAUSE_MIN = 120.0


@dataclass(frozen=True)
class EntryGuardConfig:
    """Tunable entry-only guards. Loaded from env; overridable in tests."""

    no_entry_window: tuple[time, time] | None
    no_entry_tz: str = DEFAULT_NO_ENTRY_TZ
    max_entries_per_hour: int = DEFAULT_MAX_ENTRIES_PER_HOUR
    loss_streak_pause_n: int = DEFAULT_LOSS_STREAK_PAUSE_N
    loss_streak_pause_min: float = DEFAULT_LOSS_STREAK_PAUSE_MIN


def window_label(config: EntryGuardConfig) -> str:
    """Hours the blotter can show without re-reading env."""
    window = config.no_entry_window
    if window is None:
        return "выкл"
    start, end = window
    tz = "IL" if config.no_entry_tz == DEFAULT_NO_ENTRY_TZ else config.no_entry_tz
    return f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')} {tz}"


def daily_loss_hit(day_pnl_pct: float, limit_pct: float = DAILY_LOSS_LIMIT_PCT) -> bool:
    """True when day PnL percent is at or past the loss limit (default −2.5%)."""
    return float(day_pnl_pct) <= -abs(float(limit_pct))


@dataclass(frozen=True)
class EntryGuardState:
    """Snapshot of guard inputs + derived flags for risk + bot-status.

    Caps live on the snapshot so the blotter does not reload env and disagree
    with the process that actually blocked the entry.
    """

    window_active: bool
    entries_last_hour: int
    loss_streak: int
    pause_until: datetime | None
    max_entries_per_hour: int = DEFAULT_MAX_ENTRIES_PER_HOUR
    loss_streak_pause_n: int = DEFAULT_LOSS_STREAK_PAUSE_N
    window_label: str = "03:00–09:00 IL"

    def as_dict(self) -> dict[str, object]:
        return {
            "window_active": self.window_active,
            "entries_last_hour": self.entries_last_hour,
            "loss_streak": self.loss_streak,
            "pause_until": None
            if self.pause_until is None
            else self.pause_until.astimezone(timezone.utc).isoformat(),
            "max_entries_per_hour": self.max_entries_per_hour,
            "loss_streak_pause_n": self.loss_streak_pause_n,
            "window_label": self.window_label,
        }


def _parse_hhmm(raw: str) -> time:
    parts = raw.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"expected HH:MM, got {raw!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"out-of-range HH:MM: {raw!r}")
    return time(hour, minute)


def parse_no_entry_window(raw: str | None) -> tuple[time, time] | None:
    """Parse `HH:MM-HH:MM`. Empty / off / none disables. None input → default window."""
    if raw is None:
        raw = DEFAULT_NO_ENTRY_WINDOW
    text = raw.strip()
    if not text or text.lower() in {"off", "none", "disable", "disabled", "0"}:
        return None
    if "-" not in text:
        raise ValueError(f"NO_ENTRY_WINDOW must look like HH:MM-HH:MM, got {raw!r}")
    left, right = text.split("-", 1)
    start = _parse_hhmm(left)
    end = _parse_hhmm(right)
    if start == end:
        return None
    return start, end


def in_no_entry_window(
    now: datetime,
    start: time,
    end: time,
    tz_name: str = DEFAULT_NO_ENTRY_TZ,
) -> bool:
    """True when local clock in `tz_name` sits inside [start, end). Supports wrap."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local = now.astimezone(ZoneInfo(tz_name))
    t = local.timetz().replace(tzinfo=None)
    if start < end:
        return start <= t < end
    # Wraps midnight: e.g. 22:00-06:00
    return t >= start or t < end


def loss_streak_from_closes(
    closes: list[tuple[datetime, float]],
) -> tuple[int, datetime | None]:
    """Newest-first (ts, realized_pnl). Break-even / win resets. Returns (streak, latest_loss_ts)."""
    streak = 0
    latest_loss_ts: datetime | None = None
    for ts, pnl in closes:
        if float(pnl) < 0:
            streak += 1
            if latest_loss_ts is None:
                latest_loss_ts = ts
        else:
            break
    return streak, latest_loss_ts


def load_entry_guard_config(
    environ: dict[str, str] | None = None,
) -> EntryGuardConfig:
    env = os.environ if environ is None else environ

    window_key = env.get("NO_ENTRY_WINDOW")
    # Unset → default window; explicit empty/off → disabled.
    window = parse_no_entry_window(None if window_key is None else window_key)

    tz = (env.get("NO_ENTRY_TZ") or DEFAULT_NO_ENTRY_TZ).strip() or DEFAULT_NO_ENTRY_TZ

    def _int(name: str, default: int) -> int:
        raw = env.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        return int(str(raw).strip())

    def _float(name: str, default: float) -> float:
        raw = env.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        return float(str(raw).strip())

    return EntryGuardConfig(
        no_entry_window=window,
        no_entry_tz=tz,
        max_entries_per_hour=max(0, _int("MAX_ENTRIES_PER_HOUR", DEFAULT_MAX_ENTRIES_PER_HOUR)),
        loss_streak_pause_n=max(0, _int("LOSS_STREAK_PAUSE_N", DEFAULT_LOSS_STREAK_PAUSE_N)),
        loss_streak_pause_min=max(
            0.0, _float("LOSS_STREAK_PAUSE_MIN", DEFAULT_LOSS_STREAK_PAUSE_MIN)
        ),
    )


def build_entry_guard_state(
    *,
    config: EntryGuardConfig,
    now: datetime,
    entries_last_hour: int,
    loss_streak: int,
    last_loss_ts: datetime | None,
) -> EntryGuardState:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    window_active = False
    if config.no_entry_window is not None:
        start, end = config.no_entry_window
        window_active = in_no_entry_window(now, start, end, config.no_entry_tz)

    pause_until: datetime | None = None
    if (
        config.loss_streak_pause_n > 0
        and config.loss_streak_pause_min > 0
        and loss_streak >= config.loss_streak_pause_n
        and last_loss_ts is not None
    ):
        ts = last_loss_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        candidate = ts + timedelta(minutes=float(config.loss_streak_pause_min))
        # Cooldown expiry clears the pause even if the streak has not yet been
        # reset by a winning close (entries may resume; a new loss restarts it).
        if now < candidate:
            pause_until = candidate

    return EntryGuardState(
        window_active=window_active,
        entries_last_hour=int(entries_last_hour),
        loss_streak=int(loss_streak),
        pause_until=pause_until,
        max_entries_per_hour=int(config.max_entries_per_hour),
        loss_streak_pause_n=int(config.loss_streak_pause_n),
        window_label=window_label(config),
    )


def entry_guard_skip_reason(
    config: EntryGuardConfig,
    state: EntryGuardState,
    *,
    now: datetime | None = None,
) -> str | None:
    """Return a skip_reason for a prospective buy_long, or None if allowed."""
    if config.no_entry_window is not None and state.window_active:
        return "no_entry_window"
    if config.max_entries_per_hour > 0 and state.entries_last_hour >= config.max_entries_per_hour:
        return "hourly_entry_cap"
    if config.loss_streak_pause_n > 0 and state.pause_until is not None:
        clock = now or datetime.now(timezone.utc)
        if clock.tzinfo is None:
            clock = clock.replace(tzinfo=timezone.utc)
        if clock < state.pause_until:
            return "loss_streak_pause"
    return None


def long_protective_stop(
    snapshot: MarketSnapshot,
    features: Features,
    account: AccountState,
) -> float | None:
    """Stop for an open long: the stored one, or entry minus ATR. None if flat."""
    if snapshot.position.side != "LONG" or snapshot.position.size <= 0:
        return None
    stop = snapshot.position.stop_price
    if stop is None and features.atr and features.atr > 0 and snapshot.position.entry:
        mult = min(max(account.atr_stop_mult, ATR_STOP_MULT_MIN), ATR_STOP_MULT_MAX)
        stop = float(snapshot.position.entry) - mult * float(features.atr)
    if stop is None or stop <= 0:
        return None
    return float(stop)


def _stop_price(close: float, stop_distance: float) -> float:
    return close - stop_distance


def _order_side(action: str, position_side: str) -> str:
    """Binance long-only: BUY opens a long, SELL only closes a long."""
    if action == "buy_long":
        return "BUY"
    if action == "close" and position_side == "SHORT":
        return "BUY"
    if action in {"close", "sell_short"}:
        return "SELL"
    return "BUY"


def post_only_limit_price(order_side: str, close: float, snapshot: MarketSnapshot) -> float:
    """Maker price: join the bid on BUY, the ask on SELL; else last close."""
    book = snapshot.book
    if book is not None:
        if order_side == "BUY" and book.bids:
            return book.bids[0].price
        if order_side == "SELL" and book.asks:
            return book.asks[0].price
    return close


def apply_risk(
    policy: PolicyDecision,
    features: Features,
    snapshot: MarketSnapshot,
    account: AccountState,
    *,
    seconds_since_last_close: float | None = None,
    reentry_cooldown_sec: float = REENTRY_COOLDOWN_SEC,
    seconds_since_last_entry: float | None = None,
    min_hold_sec: float = MIN_HOLD_SEC,
    entry_guard_config: EntryGuardConfig | None = None,
    entry_guard_state: EntryGuardState | None = None,
    now: datetime | None = None,
) -> TradeIntent:
    """Size = risk% / ATR-stop. Jev probabilities never enter the size formula."""
    symbol = snapshot.symbol
    close = features.close
    stop_mult = min(max(account.atr_stop_mult, ATR_STOP_MULT_MIN), ATR_STOP_MULT_MAX)

    def skip(reason: str, action: str = "hold") -> TradeIntent:
        return TradeIntent(
            action=action,
            qty=0.0,
            stop_price=None,
            stop_distance=None,
            entry_type="NONE",
            reduce_only=False,
            client_order_id=client_order_id(symbol, action),
            symbol=symbol,
            risk_pct=account.risk_pct,
            order_side=_order_side(action, snapshot.position.side),
            skip_reason=reason,
            risk_event=reason,
        )

    in_position = snapshot.position.side != "FLAT" and snapshot.position.size > 0
    pos_side = snapshot.position.side

    def close_position(*, risk_event: str | None, entry_type: str) -> TradeIntent:
        side = _order_side("close", pos_side)
        limit = None if entry_type == "MARKET" else post_only_limit_price(side, close, snapshot)
        tag = "flatten" if entry_type == "MARKET" else "close"
        return market_close_intent(
            symbol=symbol,
            qty=snapshot.position.size,
            order_side=side,
            client_order_id=client_order_id(symbol, tag),
            risk_event=risk_event,
            risk_pct=account.risk_pct,
            limit_price=limit,
            entry_type=entry_type,
        )

    if account.kill_switch:
        if in_position:
            return close_position(risk_event="kill_switch", entry_type="MARKET")
        return skip("kill_switch")

    if daily_loss_hit(account.daily_pnl_pct, account.daily_loss_limit_pct):
        if in_position:
            return close_position(risk_event="daily_loss", entry_type="MARKET")
        return skip("daily_loss")

    # Long-only: leftover shorts are flattened; SELL never opens a position.
    if in_position and pos_side == "SHORT":
        return close_position(risk_event="no_short", entry_type="MARKET")

    if in_position and pos_side == "LONG":
        stop = long_protective_stop(snapshot, features, account)
        if stop is not None and close <= stop:
            return close_position(risk_event="stop", entry_type="MARKET")

    if not policy.passed:
        return skip(policy.skip_reason or "hold")

    if policy.action == "sell_short":
        return skip("no_short")

    if policy.action == "close":
        if not in_position or pos_side != "LONG":
            return skip("flat")
        if (
            seconds_since_last_entry is not None
            and min_hold_sec > 0
            and seconds_since_last_entry < min_hold_sec
        ):
            return skip("min_hold")
        return close_position(risk_event=None, entry_type="MARKET")

    if policy.action != "buy_long":
        return skip(policy.skip_reason or "hold")

    if in_position and pos_side == "LONG":
        return skip("already_long")

    if not account.wallet_ok:
        return skip("wallet_unavailable")

    if (
        seconds_since_last_close is not None
        and reentry_cooldown_sec > 0
        and seconds_since_last_close < reentry_cooldown_sec
    ):
        return skip("reentry_cooldown")

    if account.open_positions >= account.max_positions:
        return skip("max_positions")

    if entry_guard_config is not None and entry_guard_state is not None:
        guard_skip = entry_guard_skip_reason(
            entry_guard_config, entry_guard_state, now=now
        )
        if guard_skip:
            return skip(guard_skip)

    atr = features.atr
    if atr is None or atr <= 0 or close <= 0:
        return skip("no_atr")

    stop_distance = stop_mult * atr
    risk_amount = account.equity_usdt * account.risk_pct
    qty = risk_amount / stop_distance
    available = account.available_usdt
    if available is None:
        available = account.equity_usdt
    leverage = max(float(account.leverage or 1.0), 1.0)
    max_notional = max(0.0, float(available)) * leverage
    if max_notional <= 0:
        # No free margin is a hard stop, not "no limit".
        return skip("insufficient_margin")
    if close > 0:
        qty = min(qty, max_notional / close)
    if qty <= 0:
        return skip("insufficient_margin")
    return TradeIntent(
        action="buy_long",
        qty=qty,
        stop_price=_stop_price(close, stop_distance),
        stop_distance=stop_distance,
        entry_type="LIMIT_POST_ONLY",
        reduce_only=False,
        client_order_id=client_order_id(symbol, "buy_long"),
        symbol=symbol,
        risk_pct=account.risk_pct,
        order_side="BUY",
        limit_price=post_only_limit_price("BUY", close, snapshot),
        skip_reason=None,
        risk_event=None,
    )
