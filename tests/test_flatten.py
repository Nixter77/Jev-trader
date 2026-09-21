from __future__ import annotations

from pathlib import Path

from jev_trader.cycle import run_once_from_answers
from jev_trader.execution import PaperBroker
from jev_trader.flatten import flatten_open_positions
from jev_trader.ledger import Ledger


def test_flatten_open_positions_closes_paper_long(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    run_once_from_answers(
        market_snapshot, passing_answers, broker=PaperBroker(), ledger=ledger
    )
    opened = ledger.load_position("BTCUSDT")
    assert opened.side == "LONG"
    assert opened.size > 0
    result = flatten_open_positions(PaperBroker(), ledger)
    assert result["closes"]
    assert result["closes"][0]["symbol"] == "BTCUSDT"
    assert result["closes"][0]["status"] == "paper_recorded"
    flat = ledger.load_position("BTCUSDT")
    assert flat.side == "FLAT"
    assert flat.size == 0.0
    book = ledger.book()
    actions = [row["action"] for row in book["fills"]]
    assert "close" in actions
    assert ledger.count_open_positions() == 0


def test_flatten_open_positions_noop_when_flat(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    result = flatten_open_positions(PaperBroker(), ledger)
    assert result["closes"] == []
    assert ledger.count_open_positions() == 0
