"""The protective (ATR) stop only moves up, also while ledger sync keeps failing."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from jev_trader.cycle import arm_exchange_stop, run_once
from jev_trader.execution import PaperBroker
from jev_trader.features import compute_features
from jev_trader.jev import judgment_from_dict
from jev_trader.ledger import Ledger
from jev_trader.models import AccountState
from jev_trader.risk import long_protective_stop
from tests.test_stop_swap_covered import Exchange, _broker, _stop

HOLD = {"action": "hold", "trend_aligned": 0.2, "false_break_risk": 0.2, "signal_strength": "слабый", "should_trade_now": 0.1, "model": "jev-1.13.0"}


def _ledger_long(tmp_path, symbol: str, entry: float, stop: float | None) -> Ledger:
    ledger = Ledger(tmp_path / "l.sqlite")
    with ledger._connect() as conn:
        ledger._upsert_position(
            conn, symbol=symbol, side="LONG", size=0.01, entry=entry,
            cash_usdt=1000.0, realized_pnl_usdt=0.0, last_mark=entry, stop_price=stop,
        )
        conn.commit()
    return ledger


class Recorder(PaperBroker):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[float, dict[str, Any]]] = []

    def ensure_stop_market(self, symbol: str, *, stop_price: float, order_side: str = "SELL", **kw: Any):
        self.calls.append((float(stop_price), kw))
        return {"http_status": 200, "body": {"skipped": "stop_exists"}}


def _no_stop_long(market_snapshot):
    features = compute_features(market_snapshot)
    entry = features.close
    snap = replace(
        market_snapshot,
        position=replace(market_snapshot.position, side="LONG", size=0.01, entry=entry, stop_price=None),
    )
    computed = long_protective_stop(snap, features, AccountState(equity_usdt=1000.0, daily_pnl_pct=0.0, kill_switch=False, open_positions=1))
    assert computed is not None and computed < features.close
    return snap, features, computed


def test_recomputed_lower_atr_stop_keeps_the_stored_one(tmp_path, market_snapshot) -> None:
    # Sync failed, so the snapshot has no stop and the ATR fallback is lower
    # than what the ledger remembers: the stored stop stays.
    snap, features, computed = _no_stop_long(market_snapshot)
    stored = computed + (features.close - computed) / 2
    ledger = _ledger_long(tmp_path, snap.symbol, snap.position.entry, stored)
    broker = Recorder()
    run_once(snap, judgment=judgment_from_dict(HOLD), broker=broker, ledger=ledger)
    assert broker.calls[0][0] == pytest.approx(stored)
    assert broker.calls[0][1].get("tighten_only") is True
    assert ledger.stored_stop(snap.symbol) == pytest.approx(stored)


def test_higher_recomputed_stop_tightens(tmp_path, market_snapshot) -> None:
    snap, _features, computed = _no_stop_long(market_snapshot)
    ledger = _ledger_long(tmp_path, snap.symbol, snap.position.entry, computed - 100.0)
    broker = Recorder()
    run_once(snap, judgment=judgment_from_dict(HOLD), broker=broker, ledger=ledger)
    assert broker.calls[0][0] == pytest.approx(computed)
    assert ledger.stored_stop(snap.symbol) == pytest.approx(computed)


def test_arm_tighten_only_never_lowers_the_ledger_stop(tmp_path) -> None:
    ledger = _ledger_long(tmp_path, "BTCUSDT", 100.0, 95.0)
    broker = Recorder()
    arm_exchange_stop(broker, ledger, "BTCUSDT", 90.0, None, tighten_only=True)
    assert broker.calls[-1][0] == 95.0
    assert ledger.stored_stop("BTCUSDT") == 95.0
    arm_exchange_stop(broker, ledger, "BTCUSDT", 97.0, None, tighten_only=True)
    assert ledger.stored_stop("BTCUSDT") == 97.0
    # A fresh entry's stop (not tighten_only) is set as given.
    arm_exchange_stop(broker, ledger, "BTCUSDT", 92.0, None)
    assert ledger.stored_stop("BTCUSDT") == 92.0


def test_broker_keeps_a_tighter_exchange_stop() -> None:
    ex = Exchange([_stop(1, "97.0")])
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.0, tighten_only=True)
    assert res["body"]["skipped"] == "stop_tighter"
    assert res["kept_stop_price"] == 97.0
    assert [c for c in ex.calls if c[0] in {"POST", "DELETE"}] == []


def test_broker_tightens_a_looser_exchange_stop() -> None:
    ex = Exchange([_stop(1, "93.0")])
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.0, tighten_only=True)
    assert res["http_status"] == 200
    assert [(r["stopPrice"], bool(r.get("closePosition"))) for r in ex.open] == [("95", True)]


def test_old_arm_path_still_replaces_a_higher_stop() -> None:
    # Entry arms (a new position's own stop) are not ratcheted.
    ex = Exchange([_stop(1, "97.0")])
    res = _broker(ex).ensure_stop_market("BTCUSDT", stop_price=95.0)
    assert res["http_status"] == 200
    assert [r["stopPrice"] for r in ex.open] == ["95"]
