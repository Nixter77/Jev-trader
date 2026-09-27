from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dataclasses import replace

from jev_trader.cycle import decision_payload, run_once
from jev_trader.execution import PaperBroker, bump_flatten_generation
from jev_trader.features import compute_features
from jev_trader.risk import qty_after_fee_budget
from jev_trader.jev import judgment_from_dict
from jev_trader.ledger import Ledger
from jev_trader.live import (
    FiveMinuteCloseLoop,
    begin_wallet_pull,
    finish_wallet_pull,
    force_wallet,
    make_run_cycle,
    overlay_wallet_position,
)
from jev_trader.status import read_json, write_json
from jev_trader.public_market import parse_rest_klines
from jev_trader.state import build_compact_state

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_ws_open_does_not_trade_closed_bar_fires_once_each(
    passing_answers,
) -> None:
    klines = _load("klines_5m.json")
    parsed = parse_rest_klines(klines)
    calls: list = []

    def run_cycle(snapshot):
        calls.append(snapshot)
        return run_once(
            snapshot,
            judgment=judgment_from_dict(passing_answers),
            broker=PaperBroker(),
        )

    loop = FiveMinuteCloseLoop("BTCUSDT", run_cycle=run_cycle)
    loop.seed_klines([row.candle for row in parsed[:-3]])

    assert loop.handle_ws_event(_load("kline_ws_open.json")) is None
    assert calls == []
    assert loop.invocations == 0

    first = loop.handle_ws_event(_load("kline_ws_closed.json"))
    assert first is not None
    assert len(calls) == 1
    assert loop.invocations == 1
    compact = build_compact_state(calls[0])
    assert compact.payload["symbol"] == "BTCUSDT"
    assert compact.payload["tf"] == "5m"
    assert "candles" not in compact.payload
    payload = decision_payload(first)
    assert payload["action"] in {"buy_long", "sell_short", "close", "hold"}
    assert payload["judgment"] is not None or payload["skip_reason"]
    assert payload["action"] == "buy_long"
    assert payload["intent"] is not None

    assert loop.handle_ws_event(_load("kline_ws_closed.json")) is None
    assert loop.handle_ws_event(_load("kline_ws_open.json")) is None
    assert len(calls) == 1

    second = loop.handle_ws_event(_load("kline_ws_closed_2.json"))
    assert second is not None
    assert len(calls) == 2
    assert loop.invocations == 2
    assert calls[1].candles[-1].ts != calls[0].candles[-1].ts


def test_rest_klines_fire_once_per_new_closed_bar(passing_answers) -> None:
    klines = _load("klines_5m.json")
    parsed = parse_rest_klines(klines)
    calls: list = []

    def run_cycle(snapshot):
        calls.append(snapshot)
        return run_once(
            snapshot,
            judgment=judgment_from_dict(passing_answers),
            broker=PaperBroker(),
        )

    loop = FiveMinuteCloseLoop("BTCUSDT", run_cycle=run_cycle)
    forming_close = parsed[-1].close_time
    first = loop.handle_rest_klines(klines, now_ms=forming_close)
    assert first is not None
    assert loop.invocations == 1
    assert loop.handle_rest_klines(klines, now_ms=forming_close) is None
    assert loop.invocations == 1
    second = loop.handle_rest_klines(klines, now_ms=forming_close + 1)
    assert second is not None
    assert loop.invocations == 2
    assert build_compact_state(calls[0]).payload["symbol"] == "BTCUSDT"


def test_cli_live_recorded_close_twice_consistent_shape(
    tmp_path: Path, passing_answers
) -> None:
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(passing_answers), encoding="utf-8")
    env_file = tmp_path / "empty.env"
    env_file.write_text("BINANCE_FAPI_BASE=https://testnet.binancefuture.com\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.pop("BINANCE_API_KEY", None)
    env.pop("BINANCE_API_SECRET", None)
    payloads = []
    for i in (1, 2):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "jev_trader",
                "live",
                "--venue",
                "paper",
                "--no-universe",
                "--max-cycles",
                "1",
                "--no-telegram",
                "--answers",
                str(answers_path),
                "--recorded-klines",
                str(FIXTURES / "klines_5m.json"),
                "--recorded-depth",
                str(FIXTURES / "depth_5.json"),
                "--ledger",
                str(tmp_path / f"ledger-{i}.sqlite"),
                "--env",
                str(env_file),
            ],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        payloads.append(payload)
        assert payload["action"] in {"buy_long", "sell_short", "close", "hold"}
        assert payload.get("judgment") or payload.get("skip_reason")
        assert payload["execution"]["venue"] == "paper"
    assert set(payloads[0]) == set(payloads[1])
    assert payloads[0]["action"] == payloads[1]["action"]


def test_skip_existing_rest_close_waits_for_next_bar(passing_answers) -> None:
    klines = _load("klines_5m.json")
    parsed = parse_rest_klines(klines)
    loop = FiveMinuteCloseLoop(
        "BTCUSDT",
        run_cycle=lambda snapshot: run_once(
            snapshot,
            judgment=judgment_from_dict(passing_answers),
            broker=PaperBroker(),
        ),
    )
    loop.skip_existing_close = True
    assert loop.handle_rest_klines(klines, now_ms=parsed[-1].close_time) is None
    assert loop.invocations == 0
    nxt = loop.handle_rest_klines(klines, now_ms=parsed[-1].close_time + 1)
    assert nxt is not None
    assert loop.invocations == 1


def test_failing_should_trade_now_holds_on_live_close() -> None:
    klines = _load("klines_5m.json")
    answers = {
        "action": "buy_long",
        "trend_aligned": 0.81,
        "false_break_risk": 0.22,
        "signal_strength": "сильный",
        "should_trade_now": 0.36,
        "model": "jev-1.13.0",
    }
    loop = FiveMinuteCloseLoop(
        "BTCUSDT",
        run_cycle=lambda snapshot: run_once(
            snapshot,
            judgment=judgment_from_dict(answers),
            broker=PaperBroker(),
        ),
    )
    result = loop.handle_rest_klines(klines, now_ms=parse_rest_klines(klines)[-1].close_time)
    assert result is not None
    payload = decision_payload(result)
    assert payload["action"] == "hold"
    assert payload["skip_reason"] == "should_trade_now"
    assert payload["intent"] is None
    assert payload["judgment"]["action"] == "buy_long"


def test_utc_day_rolls_the_loss_anchor_without_a_restart(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
    anchor = tmp_path / "risk-anchor.json"
    anchor.write_text(
        '{"utc_date":"2026-09-22","venue":"testnet","equity_usdt":10000}',
        encoding="utf-8",
    )

    class WalletBroker:
        def fetch_wallet(self):
            return {"equity_usdt": 2700.0, "available_usdt": 2700.0, "open_positions": 0}

        def submit(self, intent):
            return PaperBroker().submit(intent)

    box = {"risk_anchor_path": str(anchor), "venue": "testnet", "start_equity_usdt": 10000.0}
    cycle = make_run_cycle(
        judgment=judgment_from_dict(passing_answers),
        account_kwargs={"daily_pnl_pct": -0.73, "kill_switch": False, "max_positions": 5},
        broker=WalletBroker(),
        ledger=Ledger(tmp_path / "ledger.sqlite"),
        notifier=None,
        typesafe_api_key=None,
        wallet_box=box,
    )
    result = cycle(market_snapshot)
    saved = json.loads(anchor.read_text(encoding="utf-8"))
    assert saved["utc_date"] == datetime.now(timezone.utc).date().isoformat()
    assert saved["equity_usdt"] == pytest.approx(2700.0)
    assert box["start_equity_usdt"] == pytest.approx(2700.0)
    assert result.skip_reason != "daily_loss"


def test_run_cycle_sizes_from_binance_wallet_not_paper_10k(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
    class WalletBroker:
        def fetch_wallet(self):
            return {"equity_usdt": 3044.7, "available_usdt": 3044.7, "open_positions": 0}

        def submit(self, intent):
            return PaperBroker().submit(intent)

    box: dict = {}
    cycle = make_run_cycle(
        judgment=judgment_from_dict(passing_answers),
        account_kwargs={"daily_pnl_pct": 0.0, "kill_switch": False, "max_positions": 5},
        broker=WalletBroker(),
        ledger=Ledger(tmp_path / "ledger.sqlite"),
        notifier=None,
        typesafe_api_key=None,
        wallet_box=box,
    )
    result = cycle(market_snapshot)
    assert box["wallet"]["equity_usdt"] == 3044.7
    assert result.intent is not None
    features = compute_features(market_snapshot)
    risk_qty = (3044.7 * 0.005) / result.intent.stop_distance
    max_qty = (3044.7 * 3.0) / features.close
    assert result.intent.qty == pytest.approx(
        qty_after_fee_budget(
            qty=min(risk_qty, max_qty),
            price=features.close,
            risk_amount=3044.7 * 0.005,
        )
    )
    assert result.intent.qty * result.intent.stop_distance != pytest.approx(10_000.0 * 0.005)


def test_open_long_gets_exchange_stop_before_the_next_order(tmp_path: Path, market_snapshot) -> None:
    calls: list[tuple[str, float, str]] = []

    class Broker(PaperBroker):
        def ensure_stop_market(self, symbol: str, *, stop_price: float, order_side: str = "SELL"):
            calls.append((symbol, float(stop_price), order_side))
            return {"http_status": 200, "body": {"skipped": "stop_exists"}}

    snapshot = replace(
        market_snapshot,
        position=replace(
            market_snapshot.position,
            side="LONG",
            size=0.01,
            entry=50_000.0,
            stop_price=40_000.0,
        ),
    )
    result = run_once(
        snapshot,
        judgment=judgment_from_dict(
            {
                "action": "hold",
                "trend_aligned": 0.2,
                "false_break_risk": 0.2,
                "signal_strength": "слабый",
                "should_trade_now": 0.1,
                "model": "jev-1.13.0",
            }
        ),
        broker=Broker(),
        ledger=Ledger(tmp_path / "ledger.sqlite"),
    )
    assert result is not None
    assert result.action == "hold"
    assert calls == [(snapshot.symbol, 40_000.0, "SELL")]


def test_flatten_during_jev_does_not_open(market_snapshot, passing_answers) -> None:
    class Judging:
        def judge(self, _compact):
            bump_flatten_generation()
            return judgment_from_dict(passing_answers)

        def close(self) -> None:
            return None

    result = run_once(market_snapshot, jev_client=Judging(), broker=PaperBroker())
    assert result.action == "hold"
    assert result.skip_reason == "flatten"
    assert result.execution is not None
    assert result.execution.status == "skipped"


def test_filled_long_asks_for_a_stop(market_snapshot, passing_answers) -> None:
    calls: list[float] = []

    class Broker(PaperBroker):
        def ensure_stop_market(self, symbol: str, *, stop_price: float, order_side: str = "SELL"):
            calls.append(float(stop_price))
            return {"http_status": 200, "body": {"status": "NEW"}}

    result = run_once(market_snapshot, judgment=judgment_from_dict(passing_answers), broker=Broker())
    assert result.execution is not None
    assert result.intent is not None
    assert len(calls) == 1
    assert calls[0] == pytest.approx(result.intent.stop_price)
    assert result.execution.detail["stop"]["body"]["status"] == "NEW"


def test_overlay_wallet_uses_exchange_position(market_snapshot) -> None:
    long_snap = overlay_wallet_position(
        market_snapshot,
        {
            "equity_usdt": 2800.0,
            "positions": [
                {"symbol": "BTCUSDT", "side": "LONG", "size": 0.5, "entry": 80_000.0},
            ],
        },
    )
    assert long_snap.position.side == "LONG"
    assert long_snap.position.size == pytest.approx(0.5)
    assert long_snap.position.cash_usdt == pytest.approx(2800.0)
    assert long_snap.position.entry == pytest.approx(80_000.0)
    flat = overlay_wallet_position(long_snap, {"equity_usdt": 2800.0, "positions": []})
    assert flat.position.side == "FLAT"
    assert flat.position.size == 0.0
    assert flat.position.stop_price is None


def test_fetch_started_before_flatten_does_not_restore_the_position() -> None:
    box: dict = {}
    gen = begin_wallet_pull(box)
    force_wallet(box, {"equity_usdt": 1.0, "positions": []})
    published = finish_wallet_pull(
        box,
        gen,
        {"equity_usdt": 9.0, "positions": [{"symbol": "BTCUSDT", "side": "LONG", "size": 1}]},
    )
    assert published is not None
    assert published["equity_usdt"] == pytest.approx(1.0)
    assert box["wallet"]["positions"] == []
    later = begin_wallet_pull(box)
    refreshed = finish_wallet_pull(box, later, {"equity_usdt": 2.0, "positions": []})
    assert refreshed is not None
    assert refreshed["equity_usdt"] == pytest.approx(2.0)


def test_concurrent_status_writes_stay_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "bot-status.json"
    errors: list[BaseException] = []

    def writer(n: int) -> None:
        try:
            for i in range(40):
                write_json(path, {"n": n, "i": i, "pad": "x" * 4000})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    saved = read_json(path)
    assert saved is not None
    assert saved["n"] in {0, 1, 2, 3}


def test_ledger_uses_wal_so_the_blotter_can_read_during_a_write(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    with ledger._connect() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
