"""Local trade blotter: open positions, fills, Jev decisions, bot heartbeat."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jev_trader.ledger import Ledger
from jev_trader.risk import DEFAULT_MAX_ENTRIES_PER_HOUR, load_entry_guard_config
from jev_trader.status import pid_alive, read_json, read_pid, seconds_to_next_5m

STARTING_CASH_USDT = 10_000.0
STALE_AFTER_SEC = 90.0
# Same default as AccountState.daily_loss_limit_pct — daily_loss blocks when
# (equity - day_start) / day_start <= -DAILY_LOSS_LIMIT_PCT.
DAILY_LOSS_LIMIT_PCT = 0.025
IL_TZ_NAME = "Asia/Jerusalem"
# judge_ms above this (with no decision_backend in status) → likely Laya local HF.
LAYA_JUDGE_MS_HINT = 2000.0


def _parse_ts(raw: Any) -> datetime | None:
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
        return float(raw)
    except (TypeError, ValueError):
        return None


def compute_daily_loss(
    *,
    day_start_equity_usdt: Any,
    equity_usdt: Any,
    limit_pct: float = DAILY_LOSS_LIMIT_PCT,
) -> dict[str, Any]:
    """Daily loss gate from day-start equity vs current wallet equity.

    Active when day PnL percent <= -limit_pct (default 2.5% of day_start).
    Formula: day_pnl = equity - day_start; day_pnl_pct = day_pnl / day_start.
    """
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
    active = day_pnl_pct <= -limit
    return {
        "active": bool(active),
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


def _skip_reason_counts(decisions: list[Any] | None) -> dict[str, int]:
    keys = (
        "daily_loss",
        "no_entry_window",
        "hourly_entry_cap",
        "loss_streak_pause",
        "reentry_cooldown",
    )
    counts = {k: 0 for k in keys}
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


def resolve_decision_backend(status: dict[str, Any] | None) -> dict[str, Any]:
    """Prefer status.decision_backend; else process env; else judge_ms hint."""
    import os

    status = status or {}
    raw = status.get("decision_backend")
    checkpoint = status.get("laya_checkpoint")
    source = "status"
    backend: str | None = None
    note: str | None = None

    if isinstance(raw, str) and raw.strip():
        backend = raw.strip().lower()
    else:
        env_backend = (os.environ.get("DECISION_BACKEND") or "").strip().lower()
        if env_backend in {"jev", "laya"}:
            backend = env_backend
            source = "env"
            if not checkpoint and backend == "laya":
                checkpoint = (os.environ.get("LAYA_CHECKPOINT") or "").strip() or None
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
                backend = None
                source = "unknown"
                note = (
                    "не указано в статусе "
                    "(нужен рестарт бота чтобы писать decision_backend)"
                )

    if backend == "laya" and not checkpoint and source == "status":
        checkpoint = status.get("laya_checkpoint")
    if backend == "laya" and isinstance(checkpoint, str) and not checkpoint.strip():
        checkpoint = None
    if backend != "laya":
        checkpoint = checkpoint if backend == "laya" else None

    label = {
        "jev": "Jev",
        "laya": "Laya",
    }.get(backend or "", "не указано")

    return {
        "backend": backend,
        "label": label,
        "laya_checkpoint": checkpoint if backend == "laya" else None,
        "source": source,
        "note": note,
    }


def _format_il(raw: Any) -> str | None:
    dt = _parse_ts(raw)
    if dt is None:
        return None
    try:
        from zoneinfo import ZoneInfo

        local = dt.astimezone(ZoneInfo(IL_TZ_NAME))
    except Exception:  # noqa: BLE001
        local = dt
    return local.strftime("%Y-%m-%d %H:%M:%S IL")


def build_restrictions(
    status: dict[str, Any] | None,
    *,
    daily_loss: dict[str, Any],
) -> dict[str, Any]:
    """UI-ready restriction chips from entry_guards + daily_loss."""
    status = status or {}
    guards = status.get("entry_guards") if isinstance(status.get("entry_guards"), dict) else {}
    try:
        cfg = load_entry_guard_config()
        max_entries = int(cfg.max_entries_per_hour)
    except Exception:  # noqa: BLE001
        max_entries = DEFAULT_MAX_ENTRIES_PER_HOUR

    window_active = bool(guards.get("window_active"))
    entries_last_hour = int(guards.get("entries_last_hour") or 0)
    loss_streak = int(guards.get("loss_streak") or 0)
    pause_until_raw = guards.get("pause_until")
    pause_until_dt = _parse_ts(pause_until_raw)
    now = datetime.now(timezone.utc)
    pause_active = pause_until_dt is not None and now < pause_until_dt
    hourly_active = max_entries > 0 and entries_last_hour >= max_entries

    items = [
        {
            "id": "daily_loss",
            "label": "Daily loss",
            "active": bool(daily_loss.get("active")),
            "detail": daily_loss.get("detail") or "—",
        },
        {
            "id": "no_entry_window",
            "label": "Ночное окно 03–09 IL",
            "active": window_active,
            "detail": "активно — входы закрыты" if window_active else "нет (вне окна)",
        },
        {
            "id": "hourly_entry_cap",
            "label": "Лимит входов в час",
            "active": hourly_active,
            "detail": f"{entries_last_hour} / {max_entries}",
            "entries_last_hour": entries_last_hour,
            "max_entries_per_hour": max_entries,
        },
        {
            "id": "loss_streak_pause",
            "label": "Пауза после 3 убытков",
            "active": pause_active,
            "detail": (
                f"streak {loss_streak}; до {_format_il(pause_until_raw) or pause_until_raw}"
                if pause_active
                else f"нет (streak {loss_streak})"
            ),
            "loss_streak": loss_streak,
            "pause_until": pause_until_raw,
            "pause_until_il": _format_il(pause_until_raw),
        },
    ]
    return {
        "items": items,
        "entry_guards": {
            "window_active": window_active,
            "entries_last_hour": entries_last_hour,
            "loss_streak": loss_streak,
            "pause_until": pause_until_raw,
            "max_entries_per_hour": max_entries,
        },
        "any_active": any(bool(i.get("active")) for i in items),
    }


def bot_view(status: dict[str, Any] | None, *, pid_path: str | Path | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    pid = None if status is None else status.get("pid")
    if pid is None:
        pid = read_pid(pid_path)
    alive_pid = pid_alive(pid if isinstance(pid, int) else None)
    ts = None if status is None else _parse_ts(status.get("ts"))
    age = None if ts is None else (now - ts).total_seconds()
    stale = age is not None and age > STALE_AFTER_SEC
    running = bool(status) and alive_pid and not stale
    if running:
        state = "running"
    elif status and not alive_pid:
        state = "stopped"
    elif status and stale:
        state = "stale"
    else:
        state = "idle"
    watch = [] if status is None else list(status.get("watch") or [])
    return {
        "state": state,
        "running": running,
        "pid": pid,
        "ts": None if status is None else status.get("ts"),
        "age_sec": None if age is None else round(age, 1),
        "venue": None if status is None else status.get("venue"),
        "follow_jev": None if status is None else status.get("follow_jev"),
        "cycles": 0 if status is None else status.get("cycles") or 0,
        "watch": watch,
        "universe": None if status is None else status.get("universe"),
        "seconds_to_next_5m": seconds_to_next_5m(),
        "last_error": None if status is None else status.get("last_error"),
        "hint": None if status is None else status.get("hint"),
    }


def dashboard_state(
    ledger: Ledger,
    *,
    status_path: str | Path | None = None,
    pid_path: str | Path | None = None,
    limit: int = 40,
    can_flatten: bool = False,
) -> dict[str, Any]:
    book = ledger.book(limit=limit)
    status = read_json(status_path)
    bot = bot_view(status, pid_path=pid_path)
    positions = list(book.get("positions") or [])
    open_positions = [
        row
        for row in positions
        if str(row.get("side") or "FLAT") != "FLAT" and float(row.get("size") or 0.0) > 0
    ]
    realized = float(book.get("realized_pnl_usdt") or 0.0)
    unrealized = float(book.get("unrealized_pnl_usdt") or 0.0)
    wallet = None if status is None else status.get("wallet")
    starting = STARTING_CASH_USDT
    equity = STARTING_CASH_USDT + realized + unrealized
    if isinstance(wallet, dict) and wallet.get("equity_usdt") is not None:
        starting = float(wallet["equity_usdt"])
        equity = float(wallet["equity_usdt"])
        w_unreal = sum(float(p.get("unrealized_pnl_usdt") or 0.0) for p in wallet.get("positions") or [])
        if w_unreal:
            unrealized = w_unreal
        stops = {
            str(row.get("symbol") or "").upper(): row.get("stop_price")
            for row in positions
        }
        w_pos = [
            {
                "symbol": p.get("symbol"),
                "side": p.get("side"),
                "size": p.get("size"),
                "entry": p.get("entry"),
                "last_mark": None,
                "stop_price": stops.get(str(p.get("symbol") or "").upper()),
                "unrealized_pnl_usdt": p.get("unrealized_pnl_usdt"),
            }
            for p in wallet.get("positions") or []
            if p.get("side") != "FLAT" and float(p.get("size") or 0.0) > 0
        ]
        if w_pos:
            open_positions = w_pos
    day_start = None if status is None else status.get("day_start_equity_usdt")
    # Prefer live wallet equity for the daily-loss gate (same numbers risk uses).
    wallet_equity = None
    if isinstance(wallet, dict):
        wallet_equity = wallet.get("equity_usdt")
    if wallet_equity is None:
        wallet_equity = equity
    daily_loss = compute_daily_loss(
        day_start_equity_usdt=day_start,
        equity_usdt=wallet_equity,
    )
    decision = resolve_decision_backend(status)
    restrictions = build_restrictions(status, daily_loss=daily_loss)
    status_decisions = list((status or {}).get("last_decisions") or [])
    return {
        "ok": True,
        "ledger": str(ledger.path),
        "bot": bot,
        "starting_cash_usdt": starting,
        "realized_pnl_usdt": realized,
        "unrealized_pnl_usdt": unrealized,
        "equity_usdt": equity,
        "wallet": wallet,
        "day_start_equity_usdt": day_start,
        "decision_backend": decision.get("backend"),
        "laya_checkpoint": decision.get("laya_checkpoint"),
        "decision_model": decision,
        "entry_guards": restrictions.get("entry_guards"),
        "daily_loss": daily_loss,
        "restrictions": restrictions,
        "skip_reason_counts": _skip_reason_counts(status_decisions),
        "open_count": len(open_positions),
        "can_flatten": bool(can_flatten),
        "open_positions": open_positions,
        "positions": positions,
        "fills": book.get("fills") or [],
        "recent_decisions": book.get("recent_decisions") or [],
        "status_decisions": status_decisions,
    }


def render_html(state: dict[str, Any] | None = None) -> str:
    initial = json.dumps(state or {}, ensure_ascii=False, default=str).replace("<", "\\u003c")
    return MONITOR_HTML.replace("__INITIAL_STATE__", initial)


class MonitorHandler(BaseHTTPRequestHandler):
    ledger: Ledger
    status_path: Path
    pid_path: Path
    flatten_fn: Callable[[], dict[str, Any]] | None = None
    _flatten_lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _state(self) -> dict[str, Any]:
        return dashboard_state(
            self.ledger,
            status_path=self.status_path,
            pid_path=self.pid_path,
            can_flatten=callable(self.flatten_fn),
        )

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length") or "0"
        try:
            length = max(0, int(raw_len))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html", "/monitor"}:
            html = render_html(self._state()).encode("utf-8")
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path in {"/api/state", "/api/book"}:
            self._send_json(200, self._state())
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/flatten":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        payload = self._read_json_body()
        if payload.get("confirm") is not True:
            self._send_json(400, {"ok": False, "error": "confirm_required"})
            return
        flatten = self.flatten_fn
        if not callable(flatten):
            self._send_json(
                409,
                {
                    "ok": False,
                    "error": "flatten_unavailable",
                    "hint": "кнопка работает, когда монитор запущен вместе с ботом (run)",
                },
            )
            return
        if not self._flatten_lock.acquire(blocking=False):
            self._send_json(409, {"ok": False, "error": "flatten_busy"})
            return
        try:
            result = flatten()
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        finally:
            self._flatten_lock.release()
        state = self._state()
        self._send_json(
            200,
            {
                "ok": True,
                "closes": None if not isinstance(result, dict) else result.get("closes") or [],
                "cancelled": None if not isinstance(result, dict) else result.get("cancelled") or [],
                "wallet": None if not isinstance(result, dict) else result.get("wallet"),
                "open_count": state.get("open_count"),
                "state": state,
            },
        )


def make_handler(
    ledger: Ledger,
    status_path: str | Path,
    pid_path: str | Path,
    flatten_fn: Callable[[], dict[str, Any]] | None = None,
):
    class BoundHandler(MonitorHandler):
        pass

    BoundHandler.ledger = ledger
    BoundHandler.status_path = Path(status_path)
    BoundHandler.pid_path = Path(pid_path)
    # A raw function on the class would bind `self` as the first argument.
    BoundHandler.flatten_fn = staticmethod(flatten_fn) if flatten_fn is not None else None
    BoundHandler._flatten_lock = threading.Lock()
    return BoundHandler


class ReuseThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def bind_monitor(
    ledger: Ledger,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    status_path: str | Path = "data/bot-status.json",
    pid_path: str | Path = "data/bot.pid",
    flatten_fn: Callable[[], dict[str, Any]] | None = None,
) -> ThreadingHTTPServer:
    handler = make_handler(ledger, status_path, pid_path, flatten_fn=flatten_fn)
    return ReuseThreadingHTTPServer((host, port), handler)


def serve_monitor(
    ledger: Ledger,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    status_path: str | Path = "data/bot-status.json",
    pid_path: str | Path = "data/bot.pid",
    flatten_fn: Callable[[], dict[str, Any]] | None = None,
) -> ThreadingHTTPServer:
    server = bind_monitor(
        ledger,
        host=host,
        port=port,
        status_path=status_path,
        pid_path=pid_path,
        flatten_fn=flatten_fn,
    )
    server.serve_forever()
    return server


MONITOR_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Jev desk — монитор сделок</title>
  <style>
    :root {
      --ink: #102033;
      --panel: #173049;
      --rule: #3d5470;
      --blotter: #e8dcc0;
      --ticket: #f4ecd8;
      --copper: #c47a3a;
      --profit: #1f7a6c;
      --loss: #a33b32;
      --mute: #8aa0b5;
      --hold: #7a6a4a;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--ink); color: var(--blotter); }
    body {
      font-family: "Iowan Old Style", Palatino, "Palatino Linotype", "Times New Roman", serif;
      min-height: 100vh;
    }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 22px 64px; }
    header.mast {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      border-bottom: 1px solid var(--rule);
      padding-bottom: 18px;
      margin-bottom: 22px;
    }
    .house { letter-spacing: 0.22em; font-size: 13px; color: var(--copper); text-transform: uppercase; }
    h1 { margin: 4px 0 0; font-size: 34px; font-weight: 600; letter-spacing: 0.02em; }
    .sub { margin-top: 6px; color: var(--mute); font-size: 15px; }
    .clock {
      min-width: 220px;
      text-align: right;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    }
    .state {
      display: inline-block;
      border: 1px solid var(--copper);
      color: var(--copper);
      padding: 2px 8px;
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    .state.on { border-color: var(--profit); color: #8fd1c6; }
    .state.off { border-color: var(--loss); color: #e7a39d; }
    .bar {
      margin-top: 10px;
      height: 8px;
      background: #0a1826;
      border: 1px solid var(--rule);
      position: relative;
    }
    .bar > i {
      display: block;
      height: 100%;
      background: var(--copper);
      width: 0%;
    }
    .eta { margin-top: 6px; font-size: 12px; color: var(--mute); }
    .kpis {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 22px;
    }
    .kpi {
      background: var(--panel);
      padding: 14px 16px;
      border-left: 3px solid var(--copper);
    }
    .kpi label {
      display: block;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--mute);
    }
    .kpi b {
      display: block;
      margin-top: 6px;
      font-size: 26px;
      font-weight: 600;
    }
    .up { color: #8fd1c6; }
    .down { color: #e7a39d; }
    h2 {
      font-size: 18px;
      font-weight: 600;
      margin: 28px 0 10px;
      letter-spacing: 0.04em;
    }
    .pos-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: 28px 0 10px;
    }
    .pos-head h2 { margin: 0; }
    button.flatten {
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      background: var(--loss);
      color: #f4ecd8;
      border: 0;
      padding: 10px 14px;
      cursor: pointer;
    }
    button.flatten:hover:not(:disabled) { filter: brightness(1.1); }
    button.flatten:disabled { opacity: 0.4; cursor: not-allowed; }
    #flatten-msg { font-size: 13px; color: var(--mute); min-height: 1.2em; margin: 0 0 8px; }
    #flatten-msg.err { color: #e7a39d; }
    #flatten-msg.ok { color: #8fd1c6; }
    .tickets { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
    .ticket {
      background: var(--ticket);
      color: #1b140c;
      padding: 14px 14px 14px 22px;
      position: relative;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 13px;
    }
    .ticket::before {
      content: "";
      position: absolute;
      left: 0; top: 0; bottom: 0; width: 10px;
      background:
        radial-gradient(circle at 5px 10px, var(--ink) 3px, transparent 3.5px) 0 0 / 10px 16px;
    }
    .ticket .sym { font-size: 18px; font-weight: 700; }
    .ticket .side { float: right; letter-spacing: 0.12em; }
    .empty {
      color: var(--mute);
      border: 1px dashed var(--rule);
      padding: 16px;
      font-size: 15px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 12.5px;
    }
    th {
      text-align: left;
      color: var(--mute);
      font-weight: 500;
      border-bottom: 1px solid var(--rule);
      padding: 8px 6px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 11px;
    }
    td { padding: 8px 6px; border-bottom: 1px solid #1c334a; vertical-align: top; }
    .tag { letter-spacing: 0.06em; }
    .tag.hold { color: #d7c394; }
    .tag.buy_long, .tag.close { color: #8fd1c6; }
    .tag.sell_short { color: #e7a39d; }
    footer {
      margin-top: 36px;
      color: var(--mute);
      font-size: 13px;
      border-top: 1px solid var(--rule);
      padding-top: 12px;
    }
    .watch { color: var(--blotter); font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
    .model-panel {
      background: var(--panel);
      border: 1px solid var(--rule);
      border-left: 4px solid var(--copper);
      padding: 16px 18px;
      margin-bottom: 22px;
    }
    .model-panel .row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      align-items: baseline;
      margin-bottom: 12px;
    }
    .model-panel .lbl {
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--mute);
    }
    .model-panel .model-name {
      font-size: 22px;
      font-weight: 600;
      color: #f4ecd8;
    }
    .model-panel .model-note {
      font-size: 13px;
      color: var(--mute);
      margin-top: 4px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .chip {
      border: 1px solid var(--rule);
      background: #0f2436;
      padding: 8px 12px;
      min-width: 160px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 12px;
      color: var(--mute);
      opacity: 0.72;
    }
    .chip .chip-label {
      display: block;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 10px;
      margin-bottom: 4px;
    }
    .chip .chip-state { font-size: 14px; font-weight: 600; color: var(--blotter); }
    .chip .chip-detail { margin-top: 4px; font-size: 11px; color: var(--mute); }
    .chip.active {
      opacity: 1;
      border-color: var(--copper);
      background: #2a1a10;
      color: #f0d2b0;
      box-shadow: inset 0 0 0 1px rgba(196,122,58,0.35);
    }
    .chip.active .chip-state { color: #e7a39d; }
    .chip.active.loss {
      border-color: var(--loss);
      background: #2a1210;
    }
    @media (max-width: 800px) {
      .kpis { grid-template-columns: 1fr 1fr; }
      header.mast { grid-template-columns: 1fr; }
      .clock { text-align: left; }
      .pos-head { flex-wrap: wrap; }
      button.flatten { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="mast">
      <div>
        <div class="house">Jev desk · Binance USD-M</div>
        <h1>Монитор сделок</h1>
        <div class="sub" id="sub">Jev решает вход, удержание и выход. Размер и стоп считает код.</div>
      </div>
      <div class="clock">
        <div id="state" class="state">…</div>
        <div class="bar" title="до закрытия 5m"><i id="wick"></i></div>
        <div class="eta" id="eta">следующий бар</div>
      </div>
    </header>
    <section class="kpis" id="kpis"></section>
    <section class="model-panel" id="model-panel" aria-label="Модель и ограничения">
      <div class="row">
        <div>
          <div class="lbl">Модель</div>
          <div class="model-name" id="model-name">—</div>
          <div class="model-note" id="model-note"></div>
        </div>
      </div>
      <div class="lbl">Ограничения сейчас</div>
      <div class="chips" id="restriction-chips"></div>
    </section>
    <div class="pos-head">
      <h2>Открытые позиции</h2>
      <button type="button" id="flatten" class="flatten" disabled>Закрыть все</button>
    </div>
    <p id="flatten-msg"></p>
    <div id="positions"></div>
    <h2>Исполнения</h2>
    <div id="fills"></div>
    <h2>Решения Jev</h2>
    <div id="decisions"></div>
    <footer>
      Обновление каждые 3 секунды · журнал <span id="ledger"></span><br>
      Список, который сейчас судит Jev: <span class="watch" id="watch"></span>
    </footer>
  </div>
  <script>
    const initial = __INITIAL_STATE__;
    let last = initial;
    const fmt = (n, d=2) => (n === null || n === undefined || n === "") ? "—" : Number(n).toLocaleString("ru-RU", {minimumFractionDigits:d, maximumFractionDigits:d});
    const money = (n) => fmt(n, 2) + " USDT";
    const clsPnl = (n) => Number(n) > 0 ? "up" : Number(n) < 0 ? "down" : "";
    const tag = (a) => `<span class="tag ${a||""}">${a||"—"}</span>`;
    function rows(headers, body) {
      if (!body) return '<div class="empty">пока пусто</div>';
      return `<table><thead><tr>${headers.map(h=>`<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`;
    }
    function render(s) {
      last = s;
      const bot = s.bot || {};
      const st = document.getElementById("state");
      st.textContent = bot.running ? "работает" : (bot.state === "stale" ? "нет пульса" : "остановлен");
      st.className = "state " + (bot.running ? "on" : "off");
      const left = Number(bot.seconds_to_next_5m || 0);
      const pct = Math.max(0, Math.min(100, 100 - (left / 300) * 100));
      document.getElementById("wick").style.width = pct + "%";
      document.getElementById("eta").textContent = "до закрытия 5m · " + Math.floor(left/60) + ":" + String(Math.floor(left%60)).padStart(2,"0")
        + " · циклов " + (bot.cycles||0)
        + " · " + (bot.venue||"paper")
        + (bot.follow_jev ? " · по решению Jev" : " · с порогами")
        + (s.wallet && s.wallet.equity_usdt != null ? " · Binance " + fmt(s.wallet.equity_usdt,2) + " USDT" : "");
      document.getElementById("kpis").innerHTML = [
        ["Капитал", money(s.equity_usdt), ""],
        ["Реал. PnL", money(s.realized_pnl_usdt), clsPnl(s.realized_pnl_usdt)],
        ["Нереал. PnL", money(s.unrealized_pnl_usdt), clsPnl(s.unrealized_pnl_usdt)],
        ["Открыто", String(s.open_count||0), ""]
      ].map(([k,v,c]) => `<div class="kpi"><label>${k}</label><b class="${c}">${v}</b></div>`).join("");
      const dm = s.decision_model || {};
      let modelTitle = dm.label || "не указано";
      if ((dm.backend || s.decision_backend) === "laya" && (dm.laya_checkpoint || s.laya_checkpoint)) {
        modelTitle = "Laya · " + (dm.laya_checkpoint || s.laya_checkpoint);
      } else if ((dm.backend || s.decision_backend) === "jev") {
        modelTitle = "Jev";
      } else if ((dm.backend || s.decision_backend) === "laya") {
        modelTitle = "Laya";
      }
      document.getElementById("model-name").textContent = modelTitle;
      document.getElementById("model-note").textContent = dm.note || (
        dm.source === "status" ? "" :
        dm.source === "env" ? "из окружения процесса монитора" :
        dm.source === "judge_ms_hint" ? (dm.note || "") :
        (dm.note || "")
      );
      const items = ((s.restrictions || {}).items) || [];
      const chips = document.getElementById("restriction-chips");
      if (!items.length) {
        chips.innerHTML = '<div class="chip"><span class="chip-label">статус</span><span class="chip-state">нет данных</span></div>';
      } else {
        chips.innerHTML = items.map(it => {
          const active = !!it.active;
          const lossish = it.id === "daily_loss" || it.id === "loss_streak_pause";
          const state = active ? "активен" : "нет";
          const cls = "chip" + (active ? " active" : "") + (active && lossish ? " loss" : "");
          return `<div class="${cls}"><span class="chip-label">${it.label||it.id}</span><span class="chip-state">${state}</span><div class="chip-detail">${it.detail||""}</div></div>`;
        }).join("");
      }
      const opens = s.open_positions || [];
      document.getElementById("positions").innerHTML = opens.length ? `<div class="tickets">${opens.map(p => `
        <div class="ticket">
          <div><span class="sym">${p.symbol}</span><span class="side">${p.side}</span></div>
          <div>qty ${fmt(p.size,4)}</div>
          <div>вход ${fmt(p.entry,4)} · mark ${fmt(p.last_mark,4)}</div>
          <div>стоп ${p.stop_price==null?"—":fmt(p.stop_price,4)}</div>
          <div class="${clsPnl(p.unrealized_pnl_usdt)}">uPnL ${fmt(p.unrealized_pnl_usdt,2)}</div>
        </div>`).join("")}</div>` : '<div class="empty">Позиций нет. Jev пока держит. Следующий вопрос — на закрытии 5-минутной свечи.</div>';
      const fills = s.fills || [];
      document.getElementById("fills").innerHTML = fills.length ? rows(
        ["время","пара","действие","qty","цена","PnL","площадка"],
        fills.map(f => `<tr>
          <td>${(f.ts||"").replace("T"," ").slice(0,19)}</td>
          <td>${f.symbol||""}</td>
          <td>${tag(f.action)}</td>
          <td>${fmt(f.qty,4)}</td>
          <td>${fmt(f.price,4)}</td>
          <td class="${clsPnl(f.realized_pnl_usdt)}">${f.realized_pnl_usdt==null?"—":fmt(f.realized_pnl_usdt,2)}</td>
          <td>${f.venue||""}</td>
        </tr>`).join("")
      ) : '<div class="empty">Исполнений ещё не было — вход только buy_long (BUY). SELL бывает, когда закрываем лонг.</div>';
      const dec = (s.recent_decisions && s.recent_decisions.length) ? s.recent_decisions : (s.status_decisions||[]);
      document.getElementById("decisions").innerHTML = dec.length ? rows(
        ["время","пара","Jev","бот","почему нет","should","qty"],
        dec.map(d => `<tr>
          <td>${(d.ts||"").replace("T"," ").slice(0,19)}</td>
          <td>${d.symbol||""}</td>
          <td>${tag(d.judgment_action || d.jev_action)}</td>
          <td>${tag(d.action)}</td>
          <td>${d.skip_reason||"—"}</td>
          <td>${d.should_trade_now==null?"—":fmt(d.should_trade_now,2)}</td>
          <td>${d.qty==null?"—":fmt(d.qty,4)}</td>
        </tr>`).join("")
      ) : '<div class="empty">Решений пока нет. Бот спрашивает Jev на каждом закрытии 5m по ликвидным USDT-парам.</div>';
      document.getElementById("ledger").textContent = s.ledger || "";
      document.getElementById("watch").textContent = (bot.watch||[]).join("  ") || "—";
      const flattenBtn = document.getElementById("flatten");
      const can = !!s.can_flatten && (s.open_count||0) > 0;
      flattenBtn.disabled = !can;
      flattenBtn.title = s.can_flatten
        ? "MARKET-закрытие всех позиций на бирже"
        : "Кнопка доступна, когда монитор запущен вместе с ботом (run)";
    }
    render(initial);
    async function tick() {
      try {
        const r = await fetch("/api/state", {cache:"no-store"});
        if (!r.ok) return;
        render(await r.json());
      } catch (e) {}
    }
    document.getElementById("flatten").addEventListener("click", async () => {
      const btn = document.getElementById("flatten");
      const msg = document.getElementById("flatten-msg");
      const n = (last && last.open_count) || 0;
      if (!n || !last.can_flatten) return;
      const venue = (last.bot || {}).venue || "";
      const question = venue === "live"
        ? "Закрыть все позиции MARKET на Binance production? Это нельзя отменить."
        : "Закрыть все открытые позиции рыночным ордером (MARKET)? Это нельзя отменить.";
      if (!confirm(question)) return;
      btn.disabled = true;
      msg.className = "";
      msg.textContent = "закрываю…";
      try {
        const r = await fetch("/api/flatten", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({confirm: true}),
          cache: "no-store",
        });
        const data = await r.json();
        if (!data.ok) {
          msg.className = "err";
          msg.textContent = data.hint || data.error || "не закрылось";
          btn.disabled = false;
          return;
        }
        const names = (data.closes || []).map(c => c.symbol).filter(Boolean);
        msg.className = "ok";
        msg.textContent = names.length
          ? ("закрыто: " + names.join(", "))
          : "позиций на бирже уже не было";
        if (data.state) render(data.state);
        else await tick();
      } catch (e) {
        msg.className = "err";
        msg.textContent = "ошибка сети";
        btn.disabled = false;
      }
    });
    setInterval(tick, 3000);
  </script>
</body>
</html>
"""
