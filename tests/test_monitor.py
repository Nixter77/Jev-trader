from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from jev_trader.cycle import decision_payload, run_once_from_answers
from jev_trader.desk import build_restrictions, compute_daily_loss
from jev_trader.execution import PaperBroker
from jev_trader.flatten import flatten_open_positions
from jev_trader.jev import judgment_from_dict
from jev_trader.ledger import Ledger
from jev_trader.live import LiveRunner, make_run_cycle
from jev_trader.monitor import bind_monitor, dashboard_state, json_text, render_html
from jev_trader.status import read_json, resolve_day_anchor, write_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def test_decision_payload_includes_symbol(market_snapshot, passing_answers) -> None:
    result = run_once_from_answers(market_snapshot, passing_answers, broker=PaperBroker())
    payload = decision_payload(result)
    assert payload["symbol"] == "BTCUSDT"
    assert payload["action"] == "buy_long"


def test_dashboard_state_shows_fill_and_open_position(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
    ledger_path = tmp_path / "ledger.sqlite"
    ledger = Ledger(ledger_path)
    run_once_from_answers(
        market_snapshot, passing_answers, broker=PaperBroker(), ledger=ledger
    )
    status_path = tmp_path / "bot-status.json"
    write_json(
        status_path,
        {
            "ts": "2099-01-01T00:00:00+00:00",
            "pid": os.getpid(),
            "venue": "paper",
            "follow_jev": True,
            "cycles": 3,
            "watch": ["BTCUSDT", "ETHUSDT"],
            "universe": True,
        },
    )
    state = dashboard_state(ledger, status_path=status_path)
    assert state["ok"] is True
    assert state["open_count"] == 1
    assert state["open_positions"][0]["symbol"] == "BTCUSDT"
    assert state["fills"]
    assert state["bot"]["running"] is True
    assert state["bot"]["follow_jev"] is True
    html = render_html(state)
    assert "Монитор сделок" in html
    assert "BTCUSDT" in html
    assert "Закрыть все" in html
    assert "стоп" in html
    assert state["open_positions"][0]["stop_price"] is not None


def test_wallet_row_keeps_the_ledger_stop(tmp_path: Path, market_snapshot, passing_answers) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    run_once_from_answers(market_snapshot, passing_answers, broker=PaperBroker(), ledger=ledger)
    stop = ledger.book()["positions"][0]["stop_price"]
    assert stop is not None
    status_path = tmp_path / "bot-status.json"
    write_json(
        status_path,
        {
            "ts": "2099-01-01T00:00:00+00:00",
            "pid": os.getpid(),
            "venue": "testnet",
            "wallet": {
                "equity_usdt": 2800.0,
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "LONG",
                        "size": 0.01,
                        "entry": 100.0,
                        "unrealized_pnl_usdt": -2.0,
                    }
                ],
            },
        },
    )
    state = dashboard_state(ledger, status_path=status_path)
    assert state["open_positions"][0]["stop_price"] == pytest.approx(stop)


def test_day_anchor_does_not_reset_on_restart(tmp_path: Path) -> None:
    path = tmp_path / "risk-anchor.json"
    assert resolve_day_anchor(path, venue="testnet", equity=3000.0, today="2026-09-21") == pytest.approx(3000.0)
    assert resolve_day_anchor(path, venue="testnet", equity=2800.0, today="2026-09-21") == pytest.approx(3000.0)
    assert resolve_day_anchor(path, venue="testnet", equity=2800.0, today="2026-09-22") == pytest.approx(2800.0)


def test_dashboard_uses_binance_wallet_equity(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    status_path = tmp_path / "bot-status.json"
    write_json(
        status_path,
        {
            "ts": "2099-01-01T00:00:00+00:00",
            "pid": os.getpid(),
            "venue": "testnet",
            "wallet": {
                "equity_usdt": 3044.7,
                "available_usdt": 3044.7,
                "open_positions": 0,
                "positions": [],
            },
        },
    )
    state = dashboard_state(ledger, status_path=status_path)
    assert state["equity_usdt"] == pytest.approx(3044.7)
    assert state["starting_cash_usdt"] == pytest.approx(3044.7)


def test_cli_monitor_json(tmp_path: Path, market_snapshot, passing_answers) -> None:
    ledger_path = tmp_path / "ledger.sqlite"
    run_once_from_answers(
        market_snapshot,
        passing_answers,
        broker=PaperBroker(),
        ledger=Ledger(ledger_path),
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jev_trader",
            "monitor",
            "--json",
            "--ledger",
            str(ledger_path),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["open_count"] == 1
    assert payload["fills"][0]["action"] == "buy_long"


def test_live_runner_writes_status(tmp_path: Path, passing_answers) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    status_path = tmp_path / "bot-status.json"
    runner = LiveRunner(
        ["BTCUSDT"],
        run_cycle=make_run_cycle(
            judgment=judgment_from_dict(passing_answers),
            account_kwargs={"daily_pnl_pct": 0.0, "kill_switch": False, "max_positions": 5},
            broker=PaperBroker(),
            ledger=ledger,
            notifier=None,
            typesafe_api_key=None,
            follow_jev=True,
        ),
        recorded_klines=json.loads((FIXTURES / "klines_5m.json").read_text(encoding="utf-8")),
        recorded_depth=json.loads((FIXTURES / "depth_5.json").read_text(encoding="utf-8")),
        max_cycles=1,
        universe=False,
        status_path=status_path,
        venue="paper",
        follow_jev=True,
        heartbeat=False,
    )
    assert runner.run() == 0
    status = read_json(status_path)
    assert status is not None
    assert status["cycles"] >= 1
    assert status["venue"] == "paper"
    assert status["follow_jev"] is True
    assert status["last_decisions"]
    assert status["last_decisions"][0]["symbol"] == "BTCUSDT"
    assert status.get("decision_backend") == "jev"
    assert status.get("entry_guards") is not None


def test_monitor_http_serves_blotter(tmp_path: Path, market_snapshot, passing_answers) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    run_once_from_answers(
        market_snapshot, passing_answers, broker=PaperBroker(), ledger=ledger
    )
    server = bind_monitor(
        ledger,
        host="127.0.0.1",
        port=0,
        status_path=tmp_path / "bot-status.json",
        pid_path=tmp_path / "bot.pid",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["open_count"] == 1
        with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
            html = resp.read().decode("utf-8")
        assert "Монитор сделок" in html
        assert "BTCUSDT" in html
        assert "Закрыть все" in html
    finally:
        server.shutdown()
        server.server_close()


def _start_monitor(ledger: Ledger, tmp_path: Path, flatten_fn=None):
    server = bind_monitor(
        ledger,
        host="127.0.0.1",
        port=0,
        status_path=tmp_path / "bot-status.json",
        pid_path=tmp_path / "bot.pid",
        flatten_fn=flatten_fn,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_flatten_post_requires_confirm(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    run_once_from_answers(
        market_snapshot, passing_answers, broker=PaperBroker(), ledger=ledger
    )
    server = _start_monitor(ledger, tmp_path, flatten_fn=lambda: flatten_open_positions(PaperBroker(), ledger))
    try:
        port = server.server_address[1]
        req = Request(
            f"http://127.0.0.1:{port}/api/flatten",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(req, timeout=5)
            raise AssertionError("expected HTTPError")
        except HTTPError as exc:
            assert exc.code == 400
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "confirm_required"
    finally:
        server.shutdown()
        server.server_close()


def test_flatten_post_closes_paper_position(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    run_once_from_answers(
        market_snapshot, passing_answers, broker=PaperBroker(), ledger=ledger
    )
    assert ledger.count_open_positions() == 1
    server = _start_monitor(
        ledger,
        tmp_path,
        flatten_fn=lambda: flatten_open_positions(PaperBroker(), ledger),
    )
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as resp:
            before = json.loads(resp.read().decode("utf-8"))
        assert before["open_count"] == 1
        assert before["can_flatten"] is True
        req = Request(
            f"http://127.0.0.1:{port}/api/flatten",
            data=json.dumps({"confirm": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["open_count"] == 0
        assert payload["state"]["open_count"] == 0
        assert ledger.count_open_positions() == 0
        assert ledger.load_position("BTCUSDT").side == "FLAT"
    finally:
        server.shutdown()
        server.server_close()


def test_flatten_post_without_callback_is_unavailable(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    server = _start_monitor(ledger, tmp_path, flatten_fn=None)
    try:
        port = server.server_address[1]
        req = Request(
            f"http://127.0.0.1:{port}/api/flatten",
            data=json.dumps({"confirm": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(req, timeout=5)
            raise AssertionError("expected HTTPError")
        except HTTPError as exc:
            assert exc.code == 409
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "flatten_unavailable"
    finally:
        server.shutdown()
        server.server_close()


def _flatten_status(port: int, headers: dict[str, str], data: bytes | None = None) -> tuple[int, dict]:
    req = Request(
        f"http://127.0.0.1:{port}/api/flatten",
        data=data if data is not None else json.dumps({"confirm": True}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_flatten_rejects_bad_content_type_host_and_origin(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    calls: list[int] = []

    def flatten() -> dict:
        calls.append(1)
        return {"closes": [], "cancelled": [], "wallet": None}

    server = _start_monitor(ledger, tmp_path, flatten_fn=flatten)
    try:
        port = server.server_address[1]
        code, body = _flatten_status(port, {"Content-Type": "text/plain"})
        assert (code, body["error"]) == (415, "content_type_must_be_json")
        code, body = _flatten_status(port, {"Content-Type": "application/x-www-form-urlencoded"}, b"confirm=true")
        assert (code, body["error"]) == (415, "content_type_must_be_json")
        code, body = _flatten_status(
            port, {"Content-Type": "application/json", "Host": f"evil.example:{port}"}
        )
        assert (code, body["error"]) == (403, "host_not_allowed")
        code, body = _flatten_status(
            port, {"Content-Type": "application/json", "Host": "127.0.0.1:1"}
        )
        assert (code, body["error"]) == (403, "host_not_allowed")
        code, body = _flatten_status(
            port, {"Content-Type": "application/json", "Origin": "http://evil.example"}
        )
        assert (code, body["error"]) == (403, "origin_not_allowed")
        assert calls == []
        code, body = _flatten_status(
            port,
            {
                "Content-Type": "application/json; charset=utf-8",
                "Host": f"localhost:{port}",
                "Origin": f"http://localhost:{port}",
            },
        )
        assert code == 200 and body["ok"] is True
        code, body = _flatten_status(
            port, {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"}
        )
        assert code == 200
        assert calls == [1, 1]
    finally:
        server.shutdown()
        server.server_close()


def test_hourly_cap_follows_the_bot_snapshot_not_monitor_env(monkeypatch) -> None:
    monkeypatch.setenv("MAX_ENTRIES_PER_HOUR", "1")
    restrictions = build_restrictions(
        {
            "entry_guards": {
                "window_active": False,
                "entries_last_hour": 2,
                "loss_streak": 0,
                "pause_until": None,
                "max_entries_per_hour": 5,
            }
        },
        daily_loss={"active": False, "detail": "день −1.00%"},
    )
    cap = next(item for item in restrictions["items"] if item["id"] == "hourly_entry_cap")
    assert cap["active"] is False
    assert cap["max_entries_per_hour"] == 5


def test_desk_rejects_nonfinite_money_and_markup_labels() -> None:
    assert compute_daily_loss(day_start_equity_usdt=1000.0, equity_usdt=float("nan"))["known"] is False
    assert compute_daily_loss(day_start_equity_usdt=1000.0, equity_usdt=float("inf"))["known"] is False
    restrictions = build_restrictions(
        {
            "entry_guards": {
                "window_label": "<img src=x onerror=alert(1)>",
                "window_active": "false",
                "pause_until": "<script>alert(1)</script>",
            }
        },
        daily_loss={"active": False, "detail": "ok"},
    )
    window = next(item for item in restrictions["items"] if item["id"] == "no_entry_window")
    pause = next(item for item in restrictions["items"] if item["id"] == "loss_streak_pause")
    assert window["active"] is False
    assert "<" not in window["label"]
    assert window["label"] == "Ночное окно 03:00–09:00 IL"
    assert "<" not in pause["detail"]
    assert pause["pause_until"] is None
    text = json_text({"n": float("nan"), "i": float("inf")})
    assert json.loads(text) == {"n": None, "i": None}
    html = render_html({"open_positions": [{"symbol": "<img>", "side": "LONG"}]})
    assert "function esc" in html
    assert "\\u003cimg>" in html


def test_compute_daily_loss_active_and_inactive() -> None:
    inactive = compute_daily_loss(day_start_equity_usdt=1000.0, equity_usdt=980.0)
    assert inactive["known"] is True
    assert inactive["active"] is False
    assert inactive["day_pnl_pct"] == pytest.approx(-0.02)
    # Exactly at -2.5% is active (<=)
    edge = compute_daily_loss(day_start_equity_usdt=1000.0, equity_usdt=975.0)
    assert edge["active"] is True
    deep = compute_daily_loss(day_start_equity_usdt=2000.0, equity_usdt=1900.0)
    assert deep["active"] is True
    assert deep["day_pnl_usdt"] == pytest.approx(-100.0)
    unknown = compute_daily_loss(day_start_equity_usdt=None, equity_usdt=100.0)
    assert unknown["known"] is False
    assert unknown["active"] is False


def test_dashboard_exposes_backend_guards_and_daily_loss(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    status_path = tmp_path / "bot-status.json"
    write_json(
        status_path,
        {
            "ts": "2099-01-01T00:00:00+00:00",
            "pid": os.getpid(),
            "venue": "testnet",
            "decision_backend": "laya",
            "laya_checkpoint": "typed-decisions",
            "day_start_equity_usdt": 1000.0,
            "wallet": {"equity_usdt": 960.0, "positions": []},
            "entry_guards": {
                "window_active": True,
                "entries_last_hour": 2,
                "loss_streak": 3,
                "pause_until": "2099-01-01T12:00:00+00:00",
            },
            "last_decisions": [
                {"skip_reason": "daily_loss", "judge_ms": 50},
                {"skip_reason": "no_entry_window", "judge_ms": 40},
                {"skip_reason": "hourly_entry_cap", "judge_ms": 40},
                {"skip_reason": "loss_streak_pause", "judge_ms": 40},
            ],
        },
    )
    state = dashboard_state(ledger, status_path=status_path)
    assert state["decision_backend"] == "laya"
    assert state["laya_checkpoint"] == "typed-decisions"
    assert state["decision_model"]["source"] == "status"
    assert state["entry_guards"]["window_active"] is True
    assert state["entry_guards"]["entries_last_hour"] == 2
    assert state["daily_loss"]["active"] is True
    assert state["daily_loss"]["day_pnl_pct"] == pytest.approx(-0.04)
    assert state["day_start_equity_usdt"] == pytest.approx(1000.0)
    ids = {i["id"]: i for i in state["restrictions"]["items"]}
    assert ids["daily_loss"]["active"] is True
    assert ids["no_entry_window"]["active"] is True
    assert ids["hourly_entry_cap"]["active"] is True
    assert ids["loss_streak_pause"]["active"] is True
    assert state["skip_reason_counts"]["daily_loss"] == 1
    assert state["skip_reason_counts"]["loss_streak_pause"] == 1
    html = render_html(state)
    assert "Ограничения сейчас" in html
    assert "Модель" in html
    assert "restriction-chips" in html


def test_dashboard_infers_laya_from_judge_ms_when_backend_missing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    status_path = tmp_path / "bot-status.json"
    write_json(
        status_path,
        {
            "ts": "2099-01-01T00:00:00+00:00",
            "pid": os.getpid(),
            "venue": "testnet",
            "day_start_equity_usdt": 2500.0,
            "wallet": {"equity_usdt": 2480.0, "positions": []},
            "entry_guards": {
                "window_active": False,
                "entries_last_hour": 0,
                "loss_streak": 0,
                "pause_until": None,
            },
            "last_decisions": [
                {"skip_reason": "reentry_cooldown", "judge_ms": 17000},
                {"skip_reason": "reentry_cooldown", "judge_ms": 18000},
            ],
        },
    )
    # Ensure process env does not force a backend for this test.
    os.environ.pop("DECISION_BACKEND", None)
    state = dashboard_state(ledger, status_path=status_path)
    assert state["decision_backend"] == "laya"
    assert state["decision_model"]["source"] == "judge_ms_hint"
    assert "рестарт" in (state["decision_model"]["note"] or "")
    assert state["daily_loss"]["active"] is False
    ids = {i["id"]: i for i in state["restrictions"]["items"]}
    assert ids["loss_streak_pause"]["active"] is False
    assert ids["no_entry_window"]["active"] is False

