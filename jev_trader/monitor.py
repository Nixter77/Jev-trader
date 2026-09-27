"""Local trade blotter: open positions, fills, Jev decisions, bot heartbeat."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jev_trader.desk import (
    build_restrictions,
    compute_daily_loss,
    parse_ts,
    present_day_income,
    resolve_decision_backend,
    skip_reason_counts,
)
from jev_trader.ledger import Ledger
from jev_trader.status import pid_alive, read_json, read_pid, seconds_to_next_5m

STARTING_CASH_USDT = 10_000.0
STALE_AFTER_SEC = 90.0
MAX_JSON_BODY = 4096


def _json_safe(value: Any) -> Any:
    """Drop NaN/Infinity so /api/state stays real JSON and fetch().json() works."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def json_text(payload: Any) -> str:
    return json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False, default=str)



def bot_view(status: dict[str, Any] | None, *, pid_path: str | Path | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    pid = None if status is None else status.get("pid")
    if pid is None:
        pid = read_pid(pid_path)
    alive_pid = pid_alive(pid if isinstance(pid, int) else None)
    ts = None if status is None else parse_ts(status.get("ts"))
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
    day_income = present_day_income(None if status is None else status.get("day_income"))
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
        "day_income": day_income,
        "restrictions": restrictions,
        "skip_reason_counts": skip_reason_counts(status_decisions),
        "open_count": len(open_positions),
        "can_flatten": bool(can_flatten),
        "open_positions": open_positions,
        "positions": positions,
        "fills": book.get("fills") or [],
        "recent_decisions": book.get("recent_decisions") or [],
        "status_decisions": status_decisions,
    }


def render_html(state: dict[str, Any] | None = None) -> str:
    initial = json_text(state or {}).replace("<", "\\u003c")
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
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json_text(payload).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length") or "0"
        try:
            length = min(MAX_JSON_BODY, max(0, int(raw_len)))
        except (ValueError, OverflowError):
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

    def _flatten_request_refusal(self) -> tuple[int, str] | None:
        """Refuse cross-site / non-JSON posts to /api/flatten.

        A browser form or a page on another origin can't send
        application/json without a preflight, and DNS rebinding shows up as a
        foreign Host. Only 127.0.0.1:<port> / localhost:<port> are accepted.
        """
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            return 415, "content_type_must_be_json"
        port = self.server.server_address[1]
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in allowed_hosts:
            return 403, "host_not_allowed"
        origin = self.headers.get("Origin")
        if origin is not None:
            if origin.strip().lower() not in {f"http://{h}" for h in allowed_hosts}:
                return 403, "origin_not_allowed"
        return None

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/flatten":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        refusal = self._flatten_request_refusal()
        if refusal is not None:
            code, error = refusal
            self._send_json(code, {"ok": False, "error": error})
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
    <p class="sub" id="day-income"></p>
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
    function esc(value) {
      return String(value == null ? "" : value).replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      })[ch]);
    }
    const tag = (a) => {
      const name = String(a || "");
      const cls = /^[a-z0-9_]+$/i.test(name) ? name : "";
      return `<span class="tag ${cls}">${esc(name || "—")}</span>`;
    };
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
      const inc = s.day_income || {};
      const dayNet = inc.known ? inc.net_usdt : null;
      document.getElementById("kpis").innerHTML = [
        ["Капитал", money(s.equity_usdt), ""],
        ["День, биржа", inc.known ? money(dayNet) : "неизвестно", inc.known ? clsPnl(dayNet) : ""],
        ["Нереал. PnL", money(s.unrealized_pnl_usdt), clsPnl(s.unrealized_pnl_usdt)],
        ["Открыто", String(s.open_count||0), ""]
      ].map(([k,v,c]) => `<div class="kpi"><label>${k}</label><b class="${c}">${v}</b></div>`).join("");
      const dayLine = document.getElementById("day-income");
      dayLine.textContent = inc.known
        ? "День биржи: цена " + money(inc.realized_usdt)
          + " · комиссия " + money(inc.commission_usdt)
          + " · фандинг " + money(inc.funding_usdt)
          + " · нетто " + money(inc.net_usdt)
        : "День биржи: неизвестно";
      const dm = s.decision_model || {};
      document.getElementById("model-name").textContent = dm.title || "не указано";
      document.getElementById("model-note").textContent = dm.note || "";
      const items = ((s.restrictions || {}).items) || [];
      const chips = document.getElementById("restriction-chips");
      if (!items.length) {
        chips.innerHTML = '<div class="chip"><span class="chip-label">статус</span><span class="chip-state">нет данных</span></div>';
      } else {
        chips.innerHTML = items.map(it => {
          const active = !!it.active;
          const state = active ? "активен" : "нет";
          const cls = "chip" + (active ? " active" : "") + (active && it.tone === "loss" ? " loss" : "");
          return `<div class="${cls}"><span class="chip-label">${esc(it.label||it.id)}</span><span class="chip-state">${state}</span><div class="chip-detail">${esc(it.detail||"")}</div></div>`;
        }).join("");
      }
      const opens = s.open_positions || [];
      document.getElementById("positions").innerHTML = opens.length ? `<div class="tickets">${opens.map(p => `
        <div class="ticket">
          <div><span class="sym">${esc(p.symbol)}</span><span class="side">${esc(p.side)}</span></div>
          <div>qty ${fmt(p.size,4)}</div>
          <div>вход ${fmt(p.entry,4)} · mark ${fmt(p.last_mark,4)}</div>
          <div>стоп ${p.stop_price==null?"—":fmt(p.stop_price,4)}</div>
          <div class="${clsPnl(p.unrealized_pnl_usdt)}">uPnL ${fmt(p.unrealized_pnl_usdt,2)}</div>
        </div>`).join("")}</div>` : '<div class="empty">Позиций нет. Jev пока держит. Следующий вопрос — на закрытии 5-минутной свечи.</div>';
      const fills = s.fills || [];
      document.getElementById("fills").innerHTML = fills.length ? rows(
        ["время","пара","действие","qty","цена","PnL","площадка"],
        fills.map(f => `<tr>
          <td>${esc(String(f.ts||"").replace("T"," ").slice(0,19))}</td>
          <td>${esc(f.symbol||"")}</td>
          <td>${tag(f.action)}</td>
          <td>${fmt(f.qty,4)}</td>
          <td>${fmt(f.price,4)}</td>
          <td class="${clsPnl(f.realized_pnl_usdt)}">${f.realized_pnl_usdt==null?"—":fmt(f.realized_pnl_usdt,2)}</td>
          <td>${esc(f.venue||"")}</td>
        </tr>`).join("")
      ) : '<div class="empty">Исполнений ещё не было — вход только buy_long (BUY). SELL бывает, когда закрываем лонг.</div>';
      const dec = (s.recent_decisions && s.recent_decisions.length) ? s.recent_decisions : (s.status_decisions||[]);
      document.getElementById("decisions").innerHTML = dec.length ? rows(
        ["время","пара","Jev","бот","почему нет","should","qty"],
        dec.map(d => `<tr>
          <td>${esc(String(d.ts||"").replace("T"," ").slice(0,19))}</td>
          <td>${esc(d.symbol||"")}</td>
          <td>${tag(d.judgment_action || d.jev_action)}</td>
          <td>${tag(d.action)}</td>
          <td>${esc(d.skip_reason||"—")}</td>
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
    let paintGen = 0;
    async function tick() {
      const gen = ++paintGen;
      try {
        const r = await fetch("/api/state", {cache:"no-store"});
        if (!r.ok || gen !== paintGen) return;
        const body = await r.json();
        if (gen !== paintGen) return;
        render(body);
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
          if (data.error === "flatten_busy") {
            msg.textContent = "уже закрываю";
            return;
          }
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
        paintGen++;
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
