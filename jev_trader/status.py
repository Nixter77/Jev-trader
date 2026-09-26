"""On-disk heartbeat so the trade monitor can see if the bot is alive."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def seconds_to_next_5m(*, now: float | None = None) -> float:
    now_ms = int((now if now is not None else time.time()) * 1000)
    interval_ms = 300_000
    wait_ms = interval_ms - (now_ms % interval_ms)
    return round(wait_ms / 1000.0, 1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: str, **fields: Any) -> None:
    """One JSON line on stderr for errors that must not stop trading."""
    row = {"type": event, "ts": utc_now(), **fields}
    try:
        print(json.dumps(row, ensure_ascii=False, default=str), file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def resolve_day_anchor(
    path: str | Path,
    *,
    venue: str,
    equity: float,
    today: str | None = None,
) -> float:
    """Equity at the start of this UTC day. A restart must not reset the loss limit."""
    day = today or datetime.now(timezone.utc).date().isoformat()
    current = read_json(path) or {}
    saved = current.get("equity_usdt")
    if (
        current.get("utc_date") == day
        and current.get("venue") == venue
        and isinstance(saved, (int, float))
        and not isinstance(saved, bool)
        and float(saved) > 0
    ):
        return float(saved)
    if equity > 0:
        write_json(path, {"utc_date": day, "venue": venue, "equity_usdt": equity})
    return equity


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomic replace. Each writer gets its own temp file so two threads cannot truncate one `.tmp`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def write_pid(path: str | Path, pid: int | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(str(pid if pid is not None else os.getpid()), encoding="utf-8")


def read_pid(path: str | Path | None) -> int | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return int(target.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def clear_pid(path: str | Path | None) -> None:
    if not path:
        return
    target = Path(path)
    try:
        target.unlink()
    except OSError:
        pass


def other_bot_running(pid_path: str | Path) -> int | None:
    pid = read_pid(pid_path)
    if pid is None or pid == os.getpid():
        return None
    if pid_alive(pid):
        return pid
    return None
