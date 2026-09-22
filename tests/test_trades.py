from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from dataclasses import replace
from pathlib import Path

from jev_trader.cycle import run_once_from_answers
from jev_trader.execution import PaperBroker
from jev_trader.ledger import Ledger
from jev_trader.models import AccountState, CycleResult, ExecutionResult, Position, TradeIntent

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


def test_accepted_new_order_does_not_open_ledger_position(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    intent = TradeIntent(
        action="buy_long",
        qty=10.0,
        stop_price=1.0,
        stop_distance=1.0,
        entry_type="LIMIT_POST_ONLY",
        reduce_only=False,
        client_order_id="jev1_fake",
        symbol="BTCUSDT",
        risk_pct=0.005,
        order_side="BUY",
        limit_price=100.0,
    )
    execution = ExecutionResult(
        status="accepted",
        venue="binance_testnet",
        client_order_id="jev1_fake",
        reduce_only=False,
        detail={"http_status": 200, "body": {"status": "NEW", "executedQty": "0", "orderId": 1}},
    )
    ledger.record(
        CycleResult(
            action="buy_long",
            skip_reason=None,
            intent=intent,
            execution=execution,
            judgment=None,
            state_text="",
            state={"symbol": "BTCUSDT", "price": {"close": 100.0}},
        )
    )
    pos = ledger.load_position("BTCUSDT")
    assert pos.side == "FLAT"
    assert ledger.book()["fills"] == []


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


@pytest.mark.parametrize("risk_event", ["stop", "daily_loss"])
def test_filled_risk_close_is_recorded_when_model_close_policy_fails(
    tmp_path: Path, market_snapshot, risk_event: str
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    # Seed the ledger with a real paper long so close PnL/accounting has an entry.
    buy_answers = {
        "action": "buy_long",
        "trend_aligned": 0.81,
        "false_break_risk": 0.20,
        "signal_strength": "рабочий",
        "should_trade_now": 0.84,
        "model": "jev-1.13.0",
    }
    bought = run_once_from_answers(
        market_snapshot, buy_answers, broker=PaperBroker(), ledger=ledger
    )
    assert bought.action == "buy_long"
    opened = ledger.load_position("BTCUSDT")
    assert opened.side == "LONG"

    # Force the independent risk stop.  The model also says close, but too weak
    # for close asymmetry; accounting must follow the filled risk intent.
    stopped_snapshot = replace(
        market_snapshot,
        position=Position(
            side="LONG",
            size=opened.size,
            cash_usdt=opened.cash_usdt,
            entry=opened.entry,
            stop_price=(
                market_snapshot.candles[-1].close + 1_000.0
                if risk_event == "stop"
                else None
            ),
        ),
    )
    account = AccountState(
        equity_usdt=opened.cash_usdt,
        daily_pnl_pct=-0.03 if risk_event == "daily_loss" else 0.0,
        kill_switch=False,
        open_positions=1,
    )
    weak_close = {
        "action": "close",
        "trend_aligned": 0.10,
        "false_break_risk": 0.20,
        "signal_strength": "рабочий",
        "should_trade_now": 0.40,
        "action_probabilities": {"close": 0.65, "hold": 0.35},
        "model": "laya:typed-decisions",
    }
    result = run_once_from_answers(
        stopped_snapshot, weak_close, broker=PaperBroker(), ledger=ledger,
        account=account, follow_jev=True,
    )

    assert result.intent is not None
    assert result.intent.risk_event == risk_event
    assert result.execution is not None
    assert result.execution.status == "paper_recorded"
    assert result.action == "close"
    assert result.skip_reason is None
    assert result.risk_event == risk_event

    book = ledger.book()
    close_fills = [row for row in book["fills"] if row["action"] == "close"]
    assert len(close_fills) == 1
    assert close_fills[0]["client_order_id"] == result.intent.client_order_id
    assert ledger.load_position("BTCUSDT").side == "FLAT"


def test_unfilled_risk_close_does_not_override_model_policy(
    tmp_path: Path, market_snapshot
) -> None:
    """Only a real fill wins; an unfilled risk close remains hold/accounting-safe."""
    class UnfilledBroker:
        def submit(self, intent: TradeIntent) -> ExecutionResult:
            return ExecutionResult(
                status="unfilled",
                venue="test",
                client_order_id=intent.client_order_id,
                reduce_only=True,
                detail={},
            )

    ledger = Ledger(tmp_path / "ledger.sqlite")
    stopped_snapshot = replace(
        market_snapshot,
        position=Position(
            side="LONG",
            size=1.0,
            cash_usdt=10_000.0,
            entry=market_snapshot.candles[-1].close,
            stop_price=market_snapshot.candles[-1].close + 1_000.0,
        ),
    )
    weak_close = {
        "action": "close",
        "trend_aligned": 0.10,
        "false_break_risk": 0.20,
        "signal_strength": "рабочий",
        "should_trade_now": 0.40,
        "action_probabilities": {"close": 0.65, "hold": 0.35},
        "model": "laya:typed-decisions",
    }
    result = run_once_from_answers(
        stopped_snapshot, weak_close, broker=UnfilledBroker(), ledger=ledger,
        follow_jev=True,
    )
    assert result.intent is not None and result.intent.risk_event == "stop"
    assert result.action == "hold"
    assert result.skip_reason == "close_should_trade_now"
    assert ledger.book()["fills"] == []
