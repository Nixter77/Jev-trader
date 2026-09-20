from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from jev_trader.cycle import decision_payload, run_once_from_answers
from jev_trader.execution import PaperBroker
from jev_trader.ledger import Ledger
from jev_trader.snapshot import snapshot_to_jsonable
from jev_trader.telegram import TelegramNotifier

ROOT = Path(__file__).resolve().parents[1]


def test_run_once_emits_action_and_intent_or_skip(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
    result = run_once_from_answers(
        market_snapshot,
        passing_answers,
        broker=PaperBroker(),
        ledger=Ledger(tmp_path / "ledger.sqlite"),
        notifier=TelegramNotifier("", "", enabled=False),
    )
    payload = decision_payload(result)
    assert payload["action"] in {"buy_long", "sell_short", "close", "hold"}
    assert payload["intent"] is not None or payload["skip_reason"]
    if payload["action"] == "buy_long":
        assert payload["intent"]["reduce_only"] is False
        assert payload["intent"]["entry_type"] == "LIMIT_POST_ONLY"
        assert payload["intent"]["qty"] > 0
        assert payload["intent"]["limit_price"] == market_snapshot.book.bids[0].price
        assert payload["skip_reason"] is None
    assert result.execution is not None
    assert result.execution.venue == "paper"


def test_cli_once_twice_consistent_shape(
    tmp_path: Path, market_snapshot, passing_answers
) -> None:
    snapshot_path = tmp_path / "market.json"
    answers_path = tmp_path / "answers.json"
    snapshot_path.write_text(
        json.dumps(snapshot_to_jsonable(market_snapshot)), encoding="utf-8"
    )
    answers_path.write_text(json.dumps(passing_answers), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    payloads = []
    for i in (1, 2):
        ledger = tmp_path / f"ledger-{i}.sqlite"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "jev_trader",
                "once",
                "--snapshot",
                str(snapshot_path),
                "--answers",
                str(answers_path),
                "--venue",
                "paper",
                "--no-telegram",
                "--ledger",
                str(ledger),
                "--env",
                str(ROOT / ".env"),
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
        assert payload.get("intent") or payload.get("skip_reason")
        assert "started" not in proc.stdout.lower() or payload["action"]
    assert set(payloads[0]) == set(payloads[1])
    assert payloads[0]["action"] == payloads[1]["action"]


def test_ledger_directory_path_uses_sqlite_file(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    ledger = Ledger(folder)
    assert ledger.path == folder / "ledger.sqlite"
    assert ledger.path.is_file()
    trailing = Ledger(str(folder) + "/")
    assert trailing.path == folder / "ledger.sqlite"
