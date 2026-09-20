from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest

from jev_trader.cycle import decision_payload, run_once_from_answers
from jev_trader.execution import PaperBroker
from jev_trader.jev import judgment_from_dict
from jev_trader.ledger import Ledger
from jev_trader.live import LiveRunner, make_run_cycle
from jev_trader.monitor import bind_monitor, dashboard_state, render_html
from jev_trader.status import read_json, write_json

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
    finally:
        server.shutdown()
        server.server_close()
