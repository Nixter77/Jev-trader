"""Dry-run counting of scripts/jev_month_replay.py (no network, no model calls)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import jev_month_replay as mr  # noqa: E402
from jev_trader.snapshot import make_uptrend_candles  # noqa: E402


def test_dry_run_counts_one_call_per_closed_bar() -> None:
    candles = make_uptrend_candles(n=400)
    start, end = candles[300].ts, candles[350].ts
    res = mr.count_symbol("BTCUSDT", candles, start, end, window=300, equity=1000.0)
    assert res["calls"] == 50
    assert res["tokens"] is None
    assert res["chars"] > 0 and res["chars_min"] <= res["chars"] / 50 <= res["chars_max"]
    outside = sum(1 for c in candles[300:350] if not mr.in_no_entry_window(c.ts))
    assert res["calls_outside_window"] == outside


def test_flat_request_offers_only_buy_or_hold() -> None:
    candles = tuple(make_uptrend_candles(n=300))
    compact, payload, body = mr.build_request("BTCUSDT", candles, 1000.0)
    assert "pos=FLAT" in compact.as_text()
    data = json.loads(body)
    assert set(data["questions"]["action"]["criteria"]) == {"buy_long", "hold"}
    assert data["model"] == payload["model"]


def test_live_needs_explicit_approval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*_a, **_k):
        raise AssertionError("model client must not be created")

    monkeypatch.setattr("jev_trader.jev.JevClient", boom)
    monkeypatch.setattr(mr, "run_live", boom)
    with pytest.raises(SystemExit):
        mr.main(["--live", "--symbols", "BTCUSDT", "--out", str(tmp_path), "--cache", str(tmp_path)])
    with pytest.raises(SystemExit):
        mr.main(["--live", "--i-have-approval", "--symbols", "BTCUSDT", "--out", str(tmp_path), "--cache", str(tmp_path)])
