from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from jev_trader.cycle import run_once_from_answers
from jev_trader.execution import PaperBroker
from jev_trader.ledger import Ledger
from jev_trader.models import Position

ROOT = Path(__file__).resolve().parents[1]


def test_sell_short_flat_does_not_fill(tmp_path: Path, market_snapshot) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    answers = {
        "action": "sell_short",
        "trend_aligned": 0.8,
        "false_break_risk": 0.2,
        "signal_strength": "сильный",
        "should_trade_now": 0.9,
        "model": "jev-1.13.0",
    }
    result = run_once_from_answers(
        market_snapshot, answers, broker=PaperBroker(), ledger=ledger
    )
    assert result.action == "hold"
    assert result.skip_reason == "no_short"
    assert ledger.book()["fills"] == []
    assert ledger.load_position("BTCUSDT").side == "FLAT"


def test_hold_does_not_open_a_fill(tmp_path: Path, market_snapshot) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    answers = {
        "action": "buy_long",
        "trend_aligned": 0.8,
        "false_break_risk": 0.2,
        "signal_strength": "сильный",
        "should_trade_now": 0.36,
        "model": "jev-1.13.0",
    }
    result = run_once_from_answers(
        market_snapshot, answers, broker=PaperBroker(), ledger=ledger
    )
    assert result.action == "hold"
    book = ledger.book()
    assert book["fills"] == []
    pos = ledger.load_position("BTCUSDT")
    assert pos.side == "FLAT"
    assert pos.size == 0.0


def test_paper_buy_then_close_records_realized_pnl(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    buy = run_once_from_answers(
        market_snapshot, passing_answers, broker=PaperBroker(), ledger=ledger
    )
    assert buy.action == "buy_long"
    assert buy.execution is not None
    assert buy.execution.status == "paper_recorded"
    opened = ledger.load_position("BTCUSDT")
    assert opened.side == "LONG"
    assert opened.size > 0
    assert opened.entry is not None
    long_snap = replace(
        market_snapshot,
        position=Position(
            side="LONG",
            size=opened.size,
            cash_usdt=opened.cash_usdt,
            entry=opened.entry,
        ),
    )
    close_answers = dict(passing_answers)
    close_answers["action"] = "close"
    closed = run_once_from_answers(
        long_snap, close_answers, broker=PaperBroker(), ledger=ledger
    )
    assert closed.action == "close"
    flat = ledger.load_position("BTCUSDT")
    assert flat.side == "FLAT"
    assert flat.size == 0.0
    book = ledger.book()
    actions = [row["action"] for row in book["fills"]]
    assert "buy_long" in actions
    assert "close" in actions
    close_fill = next(row for row in book["fills"] if row["action"] == "close")
    assert close_fill["realized_pnl_usdt"] is not None
    assert book["realized_pnl_usdt"] == close_fill["realized_pnl_usdt"]
    assert abs(flat.cash_usdt - (10_000.0 + float(close_fill["realized_pnl_usdt"]))) < 1e-6


def test_cli_trades_shows_position_after_paper_buy(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
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
            "trades",
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
    btc = next(p for p in payload["positions"] if p["symbol"] == "BTCUSDT")
    assert btc["side"] == "LONG"
    assert btc["size"] > 0
    assert payload["fills"]
    assert payload["fills"][0]["action"] == "buy_long"
    assert any(d["action"] == "buy_long" for d in payload["recent_decisions"])
