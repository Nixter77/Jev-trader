"""Moving a stopped run to a smaller plan without re-paying, prior spend, and the continuous projection guard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import jev_month_replay as mr  # noqa: E402
from jev_trader.snapshot import make_uptrend_candles  # noqa: E402

from test_month_replay_parallel import FakeJev  # noqa: E402

SYMS = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
END, DAYS = "2024-09-19", "1"


def _candles():
    return {s: make_uptrend_candles(n=110, seed=k) for k, s in enumerate(SYMS, 1)}


def _plan(candles, syms):
    sub = {s: candles[s] for s in syms}
    start = candles["AAAUSDT"][0].ts
    return mr.build_plan(sub, start, candles["AAAUSDT"][-1].ts + mr.BAR_MS)


def _meta(**kw):
    m = {"range_utc": ["2024-09-18 00:00 UTC", "2024-09-19 00:00 UTC"], "candle_venue": "mainnet", "window": 300,
         "equity": 3044.35}
    m.update(kw)
    return m


def _old_run(tmp_path: Path, candles) -> Path:
    """A stopped 3-symbol run: 90 answers, 2 billed-unparsable, a duplicate line and a torn line."""
    old = tmp_path / "old" / "live_answers.jsonl"
    old.parent.mkdir()
    plan3 = _plan(candles, SYMS)
    g = mr.BudgetGuard(4.0, 0.042, planned_calls=len(plan3), max_calls=92, check_after=10_000)
    fake = FakeJev(seed=3, billed_every=45, input_tokens=1200)
    s = mr.run_live(plan3, candles, fake, old, g, window=300, equity=3044.35, workers=3, min_interval=0.0,
                    backoff_base=0.0, max_consecutive_errors=1000, log=lambda _m: None)
    assert "max_calls" in s["stop_reason"]
    lines = old.read_text().splitlines()
    with old.open("a") as fh:
        fh.write(lines[0] + "\n")  # duplicate
        fh.write('{"symbol": "AAAUSDT", "bar_ms": 1, "ju')  # torn
    old.with_suffix(".meta.json").write_text(json.dumps(_meta(symbols=list(SYMS))))
    return old


def _snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_import_copies_only_in_plan_records_and_counts_prior_spend(tmp_path: Path) -> None:
    candles = _candles()
    old = _old_run(tmp_path, candles)
    before = _snapshot(old.parent)
    _, old_recs = mr.load_done(old)
    old_billed = mr.load_billed_errors(old.with_suffix(".errors.jsonl"))
    assert len(old_recs) + len(old_billed) == 92 and len(old_billed) >= 1
    total_old = sum(mr.charged_tokens(r) for r in old_recs + old_billed) / 1e6 * 0.042

    plan2 = _plan(candles, ("AAAUSDT", "BBBUSDT"))
    keys2 = {(p[1], p[0]) for p in plan2}
    new = tmp_path / "new" / "live_answers.jsonl"
    info = mr.import_prior(old, new, keys2, _meta(symbols=["AAAUSDT", "BBBUSDT"]), 0.042)

    assert _snapshot(old.parent) == before  # the original files are untouched
    _, new_recs = mr.load_done(new)
    new_keys = [(r["symbol"], r["bar_ms"]) for r in new_recs]
    raw_lines = [json.loads(x) for x in new.read_text().splitlines()]
    assert len(raw_lines) == len(new_recs) == len(set(new_keys))  # no duplicates, no torn lines
    assert set(new_keys) <= keys2 and all(k[0] != "CCCUSDT" for k in new_keys)
    assert set(new_keys) == {(r["symbol"], r["bar_ms"]) for r in old_recs if r["symbol"] != "CCCUSDT"}
    nb = mr.load_billed_errors(new.with_suffix(".errors.jsonl"))
    assert {(r["symbol"], r["bar_ms"]) for r in nb} == {(r["symbol"], r["bar_ms"]) for r in old_billed
                                                        if r["symbol"] != "CCCUSDT"}
    assert info["imported_answers"] == len(new_recs) and info["imported_billed_unparsable"] == len(nb)
    assert abs(info["prior_total_spend_usd"] - total_old) < 1e-6

    g = mr.BudgetGuard(4.0, 0.042, planned_calls=len(plan2), max_calls=10_000, prior_usd=info["prior_spend_usd"])
    mr.seed_guard(g, new)
    assert abs(g.total_cost() - total_old) < 1e-9  # ALL prior paid calls count against the cap exactly once
    assert g.calls == len(new_recs) + len(nb)

    fake = FakeJev(seed=4, input_tokens=1200)  # resume on the new plan: only the missing bars are asked
    s = mr.run_live(plan2, candles, fake, new, g, window=300, equity=3044.35, workers=3, min_interval=0.0,
                    backoff_base=0.0, log=lambda _m: None)
    assert s["stop_reason"] == "plan complete"
    assert sum(fake.requests.values()) == len(plan2) - len(new_recs) - len(nb)
    done, _ = mr.load_done(new)
    assert done | {(r["symbol"], r["bar_ms"]) for r in nb} == keys2
    assert _snapshot(old.parent) == before


def test_import_refuses_mismatch_and_existing_file(tmp_path: Path) -> None:
    candles = _candles()
    old = _old_run(tmp_path, candles)
    keys = {(p[1], p[0]) for p in _plan(candles, ("AAAUSDT",))}
    with pytest.raises(SystemExit, match="window"):
        mr.import_prior(old, tmp_path / "n1.jsonl", keys, _meta(window=239), 0.042)
    busy = tmp_path / "n2.jsonl"
    busy.write_text('{"x": 1}\n')
    with pytest.raises(SystemExit, match="already has records"):
        mr.import_prior(old, busy, keys, _meta(), 0.042)


def test_main_import_only_is_idempotent_and_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candles = _candles()
    old = _old_run(tmp_path, candles)
    old.with_suffix(".meta.json").write_text(json.dumps(_meta(symbols=list(SYMS))))
    monkeypatch.setattr(mr, "load_candles", lambda venue, s, a, b, cache: candles[s])
    new = tmp_path / "m9" / "live_answers.jsonl"
    argv = ["--import-only", "--symbols", "AAAUSDT,BBBUSDT", "--end", END, "--days", DAYS, "--import-from", str(old),
            "--answers", str(new), "--out", str(tmp_path / "m9"), "--max-calls", "122", "--budget-usd", "4"]
    r1 = mr.main(argv)
    assert r1["planned_calls"] == 122 and r1["import"]["imported_answers"] > 0
    assert abs(r1["spent_total_usd"] - r1["import"]["prior_total_spend_usd"]) < 1e-6
    assert r1["projection_total_usd"] < 4.0 and r1["headroom_usd"] > 0
    meta = json.loads(new.with_suffix(".meta.json").read_text())
    assert meta["import"]["source_sha256"] == mr.sha256_of(old) and len(meta["decisions"]) == len(mr.PRE_RUN_DECISIONS)
    r2 = mr.main(argv)  # a second start does not import again
    assert r2["in_file_calls"] == r1["in_file_calls"] and r2["spent_total_usd"] == r1["spent_total_usd"]
    with pytest.raises(SystemExit, match="differs"):
        mr.main([a if a != str(old) else str(tmp_path / "other.jsonl") for a in argv])
    with pytest.raises(SystemExit, match="symbols"):
        mr.main([a if a != "AAAUSDT,BBBUSDT" else "AAAUSDT" for a in argv])


def test_projection_counts_prior_spend() -> None:
    ok = mr.BudgetGuard(4.0, 0.042, planned_calls=1000, max_calls=10_000, check_after=10, prior_usd=3.9)
    for _ in range(10):
        ok.add(1000, 0, fallback_input=0)  # 1000 x $0.000042 = $0.042 for the plan
    assert abs(ok.projection() - 3.942) < 1e-9 and ok.stop_reason() is None
    over = mr.BudgetGuard(4.0, 0.042, planned_calls=1000, max_calls=10_000, check_after=10, prior_usd=3.97)
    for _ in range(10):
        over.add(1000, 0, fallback_input=0)
    assert "projection" in over.stop_reason() and "prior $3.9700" in over.stop_reason()
    hard = mr.BudgetGuard(4.0, 0.042, planned_calls=10, max_calls=10_000, prior_usd=3.99999)
    hard.add(1000, 0, fallback_input=0)
    assert "budget" in hard.stop_reason()


def test_projection_rechecked_continuously_and_catches_drift(tmp_path: Path) -> None:
    g = mr.BudgetGuard(0.5, 0.042, planned_calls=10_000, max_calls=10_000, check_after=100, recent_window=50)
    for _ in range(100):
        g.add(1000, 0, fallback_input=0)  # projection 10k x $0.000042 = $0.42 < $0.50
    assert g.stop_reason() is None
    n = 0
    while g.stop_reason() is None and n < 1000:
        g.add(1300, 0, fallback_input=0)  # drift +30 %: projection -> ~$0.55
        n += 1
    assert "projection" in g.stop_reason()
    assert n <= 50  # caught within the recent window, long before the budget is near
    assert g.total_cost() < 0.01

    # the same through run_live with workers: tokens/call jump after the first check
    candles = _candles()
    plan = _plan(candles, SYMS)

    class Drift(FakeJev):
        def judge(self, compact):  # noqa: ANN001
            raw, usage, model = super().judge(compact)
            with self.lock:
                done = sum(self.answers.values())
            usage = dict(usage, input_tokens=1000 if done <= 40 else 1400)
            return raw, usage, model

    fake = Drift(seed=5)
    budget = len(plan) * 1100 / 1e6 * 0.042  # fits 1000 tok/call, not 1400
    g2 = mr.BudgetGuard(budget, 0.042, planned_calls=len(plan), max_calls=10_000, check_after=40, recent_window=20)
    s = mr.run_live(plan, candles, fake, tmp_path / "d.jsonl", g2, window=300, equity=3044.35, workers=4,
                    min_interval=0.0, backoff_base=0.0, log=lambda _m: None)
    assert "projection" in s["stop_reason"] and g2.calls < 80 and g2.total_cost() < budget
