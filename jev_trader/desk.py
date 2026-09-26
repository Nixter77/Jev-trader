"""Blotter facts derived from bot status: model, day-loss, entry restrictions.

Pure. The HTTP server and the HTML stay in monitor.py.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jev_trader.models import DAILY_LOSS_LIMIT_PCT
from jev_trader.risk import (
    DEFAULT_LOSS_STREAK_PAUSE_N,
    DEFAULT_MAX_ENTRIES_PER_HOUR,
    daily_loss_hit,
)

IL_TZ_NAME = "Asia/Jerusalem"
# Old processes omit decision_backend. A local Laya pass is seconds, not the
# ~0.8s Jev HTTP call. Used only when status and env both omit the backend.
LAYA_JUDGE_MS_HINT = 2000.0
SKIP_REASONS = (
    "kill_switch",
    "daily_loss",
    "no_entry_window",
    "hourly_entry_cap",
    "loss_streak_pause",
    "reentry_cooldown",
)
_BACKENDS = {"jev": "Jev", "laya": "Laya"}
_LABEL_RE = re.compile(r"[\w:./ –-]{1,48}\Z")
_CHECKPOINT_RE = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
_INT_LIMIT = 1_000_000
_DEFAULT_WINDOW_LABEL = "03:00–09:00 IL"


def parse_ts(raw: Any) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    text = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _as_float(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _as_int(raw: Any, default: int) -> int:
    if raw is None or isinstance(raw, bool):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    if abs(value) > _INT_LIMIT:
        return default
    return value


def _safe_label(raw: Any, fallback: str) -> str:
    """Status text that is later painted into the page. Reject markup."""
    if not isinstance(raw, str):
        return fallback
    text = " ".join(raw.split())
    if _LABEL_RE.fullmatch(text):
        return text
    return fallback


def compute_daily_loss(
    *,
    day_start_equity_usdt: Any,
    equity_usdt: Any,
    limit_pct: float = DAILY_LOSS_LIMIT_PCT,
) -> dict[str, Any]:
    """Day-loss chip. Active under the same predicate as apply_risk."""
    start = _as_float(day_start_equity_usdt)
    equity = _as_float(equity_usdt)
    limit = abs(float(limit_pct))
    if start is None or start <= 0 or equity is None:
        return {
            "active": False,
            "known": False,
            "limit_pct": limit,
            "day_start_equity_usdt": start,
            "equity_usdt": equity,
            "day_pnl_usdt": None,
            "day_pnl_pct": None,
            "detail": "нет day_start / equity",
        }
    day_pnl = equity - start
    day_pnl_pct = day_pnl / start
    active = daily_loss_hit(day_pnl_pct, limit)
    return {
        "active": active,
        "known": True,
        "limit_pct": limit,
        "day_start_equity_usdt": start,
        "equity_usdt": equity,
        "day_pnl_usdt": round(day_pnl, 4),
        "day_pnl_pct": round(day_pnl_pct, 6),
        "detail": (
            f"день {day_pnl_pct * 100:.2f}% "
            f"(лимит −{limit * 100:.1f}%; "
            f"PnL {day_pnl:+.2f} USDT)"
        ),
    }


def skip_reason_counts(decisions: list[Any] | None) -> dict[str, int]:
    counts = {key: 0 for key in SKIP_REASONS}
    for row in decisions or []:
        if not isinstance(row, dict):
            continue
        reason = row.get("skip_reason")
        if isinstance(reason, str) and reason in counts:
            counts[reason] += 1
    return counts


def _median_judge_ms(decisions: list[Any] | None) -> float | None:
    vals: list[float] = []
    for row in decisions or []:
        if not isinstance(row, dict):
            continue
        ms = _as_float(row.get("judge_ms"))
        if ms is not None and ms > 0:
            vals.append(ms)
    if not vals:
        return None
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _checkpoint(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if _CHECKPOINT_RE.fullmatch(text):
        return text
    return None


def _model_title(backend: str | None, checkpoint: str | None) -> str:
    if backend == "laya" and checkpoint:
        return f"Laya · {checkpoint}"
    return _BACKENDS.get(backend or "", "не указано")


def resolve_decision_backend(
    status: dict[str, Any] | None,
    *,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Status field, then this process env, then a judge_ms hint for an old bot."""
    env = os.environ if environ is None else environ
    status = status or {}
    raw = status.get("decision_backend")
    backend = raw.strip().lower() if isinstance(raw, str) and raw.strip() else None
    checkpoint = _checkpoint(status.get("laya_checkpoint"))
    source = "status"
    note: str | None = None

    if backend is None:
        env_backend = (env.get("DECISION_BACKEND") or "").strip().lower()
        if env_backend in _BACKENDS:
            backend = env_backend
            source = "env"
            if backend == "laya" and checkpoint is None:
                checkpoint = _checkpoint(env.get("LAYA_CHECKPOINT"))
            note = "из окружения процесса монитора"
        else:
            med = _median_judge_ms(status.get("last_decisions"))
            if med is not None and med >= LAYA_JUDGE_MS_HINT:
                backend = "laya"
                source = "judge_ms_hint"
                note = (
                    f"не указано в статусе; по judge_ms≈{med:.0f} мс похоже на Laya "
                    "(нужен рестарт бота чтобы писать decision_backend)"
                )
            else:
                source = "unknown"
                note = (
                    "не указано в статусе "
                    "(нужен рестарт бота чтобы писать decision_backend)"
                )

    if backend != "laya":
        checkpoint = None
    return {
        "backend": backend,
        "label": _BACKENDS.get(backend or "", "не указано"),
        "title": _model_title(backend, checkpoint),
        "laya_checkpoint": checkpoint,
        "source": source,
        "note": note,
    }


def _format_il(raw: Any) -> str | None:
    dt = parse_ts(raw)
    if dt is None:
        return None
    try:
        local = dt.astimezone(ZoneInfo(IL_TZ_NAME))
    except ZoneInfoNotFoundError:
        local = dt
    return local.strftime("%Y-%m-%d %H:%M:%S IL")


def _chip(
    *,
    chip_id: str,
    label: str,
    active: bool,
    detail: str,
    tone: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": chip_id,
        "label": label,
        "active": active,
        "detail": detail,
    }
    if tone:
        item["tone"] = tone
    if extra:
        item.update(extra)
    return item


def build_restrictions(
    status: dict[str, Any] | None,
    *,
    daily_loss: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Chips from the guard snapshot the bot wrote, plus the day-loss gate."""
    status = status or {}
    guards = status.get("entry_guards") if isinstance(status.get("entry_guards"), dict) else {}
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)

    max_entries = _as_int(guards.get("max_entries_per_hour"), DEFAULT_MAX_ENTRIES_PER_HOUR)
    pause_n = _as_int(guards.get("loss_streak_pause_n"), DEFAULT_LOSS_STREAK_PAUSE_N)
    window_text = _safe_label(guards.get("window_label"), _DEFAULT_WINDOW_LABEL)
    entries_last_hour = _as_int(guards.get("entries_last_hour"), 0)
    loss_streak = _as_int(guards.get("loss_streak"), 0)
    pause_until_raw = guards.get("pause_until")
    pause_until_dt = parse_ts(pause_until_raw)
    pause_until_iso = (
        None if pause_until_dt is None else pause_until_dt.astimezone(timezone.utc).isoformat()
    )
    pause_active = pause_until_dt is not None and clock < pause_until_dt
    window_active = guards.get("window_active") is True
    hourly_active = max_entries > 0 and entries_last_hour >= max_entries
    pause_label = (
        "Пауза после серии убытков" if pause_n <= 0 else f"Пауза после {pause_n} убытков"
    )

    items = [
        _chip(
            chip_id="daily_loss",
            label="Daily loss",
            active=bool(daily_loss.get("active")),
            detail=str(daily_loss.get("detail") or "—"),
            tone="loss",
        ),
        _chip(
            chip_id="no_entry_window",
            label=f"Ночное окно {window_text}",
            active=window_active,
            detail="активно — входы закрыты" if window_active else "нет (вне окна)",
        ),
        _chip(
            chip_id="hourly_entry_cap",
            label="Лимит входов в час",
            active=hourly_active,
            detail=f"{entries_last_hour} / {max_entries}",
            extra={
                "entries_last_hour": entries_last_hour,
                "max_entries_per_hour": max_entries,
            },
        ),
        _chip(
            chip_id="loss_streak_pause",
            label=pause_label,
            active=pause_active,
            detail=(
                f"streak {loss_streak}; до {_format_il(pause_until_iso)}"
                if pause_active
                else f"нет (streak {loss_streak})"
            ),
            tone="loss",
            extra={
                "loss_streak": loss_streak,
                "pause_until": pause_until_iso,
                "pause_until_il": _format_il(pause_until_iso),
            },
        ),
    ]
    return {
        "items": items,
        "entry_guards": {
            "window_active": window_active,
            "entries_last_hour": entries_last_hour,
            "loss_streak": loss_streak,
            "pause_until": pause_until_iso,
            "max_entries_per_hour": max_entries,
            "loss_streak_pause_n": pause_n,
            "window_label": window_text,
        },
        "any_active": any(bool(item["active"]) for item in items),
    }
