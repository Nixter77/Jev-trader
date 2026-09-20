from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_trader.snapshot import build_fixture_snapshot, snapshot_to_jsonable

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    return FIXTURES


@pytest.fixture(scope="session")
def market_snapshot(fixtures_dir: Path):
    snapshot = build_fixture_snapshot()
    path = fixtures_dir / "market.json"
    path.write_text(json.dumps(snapshot_to_jsonable(snapshot), indent=2), encoding="utf-8")
    return snapshot


@pytest.fixture
def passing_answers() -> dict:
    return {
        "action": "buy_long",
        "trend_aligned": 0.81,
        "false_break_risk": 0.22,
        "signal_strength": "рабочий",
        "should_trade_now": 0.84,
        "model": "jev-1.13.0",
    }
