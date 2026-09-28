"""Budget guard, resume and error handling of jev_month_replay's live path, with a fake client (no network)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import jev_month_replay as mr  # noqa: E402
from jev_trader.snapshot import make_uptrend_candles  # noqa: E402

RAW = {"action": "hold", "trend_aligned": 0.5, "false_break_risk": 0.5, "should_trade_now": 0.3,
       "signal_strength": "слабый", "model": "jev-1.13.0"}


class FakeClient:
    def __init__(self, input_tokens: int | None = 1000, fail: Exception | None = None) -> None:
        self.calls = 0
        self.input_tokens = input_tokens
        self.fail = fail

    def judge(self, compact):  # noqa: ANN001
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return dict(RAW), {"input_tokens": self.input_tokens, "output_tokens": 12}, "jev-1.13.0"


def _setup(n: int = 60):
    candles = {"AAAUSDT": make_uptrend_candles(n=n, seed=1), "BBBUSDT": make_uptrend_candles(n=n, seed=2)}
    start = candles["AAAUSDT"][0].ts
    end = candles["AAAUSDT"][-1].ts + mr.BAR_MS
    return candles, mr.build_plan(candles, start, end)


def _guard(plan_len: int, **kw) -> mr.BudgetGuard:
    base = dict(budget_usd=4.0, price_per_mtok=0.042, planned_calls=plan_len, max_calls=10_000, check_after=2000)
    base.update(kw)
    return mr.BudgetGuard(**base)


def test_projection_stops_only_when_over_budget() -> None:
    g = _guard(100_000, check_after=10)
    for _ in range(10):
        g.add(1000, 10, fallback_input=0)  # 1000 tok x 0.042/1M = $0.000042/call -> $4.20 for 100k calls
    assert abs(g.projection() - 4.2) < 1e-9
    assert "projection" in g.stop_reason()
    ok = _guard(100_000, check_after=10)
    for _ in range(10):
        ok.add(900, 10, fallback_input=0)  # -> $3.78
    assert ok.stop_reason() is None
    early = _guard(100_000, check_after=2000)
    for _ in range(10):
        early.add(1000, 10, fallback_input=0)
    assert early.stop_reason() is None  # not checked before check_after calls


def test_hard_stops_budget_next_call_and_max_calls() -> None:
    g = _guard(10, budget_usd=0.0001)
    g.add(1000, 0, fallback_input=0)  # $0.000042
    assert g.stop_reason() is None
    g.add(1000, 0, fallback_input=0)  # $0.000084: one more would pass $0.0001
    assert "next call" in g.stop_reason()
    m = _guard(10, max_calls=2)
    m.add(1, 0, fallback_input=0)
    m.add(1, 0, fallback_input=0)
    assert "max_calls" in m.stop_reason()
    u = _guard(10)
    for _ in range(21):
        u.add(None, None, fallback_input=5)
    assert u.input_tokens == 105 and "usage missing" in u.stop_reason()


def test_resume_skips_answered_bars_and_charges_them(tmp_path: Path) -> None:
    candles, plan = _setup()
    assert len(plan) == 22  # 11 bars x 2 symbols with >= 50 bars of history
    answers = tmp_path / "a.jsonl"
    c1 = FakeClient()
    s1 = mr.run_live(plan, candles, c1, answers, _guard(len(plan), max_calls=5), window=300, equity=1000.0,
                     min_interval=0.0, sleep=lambda _s: None, log=lambda _m: None)
    assert c1.calls == 5 and "max_calls" in s1["stop_reason"]
    with answers.open("a") as fh:
        fh.write('{"symbol": "AAAUSDT", "bar_ms": 1, "judg')  # torn line from a crash
    g2 = _guard(len(plan), max_calls=100)
    mr.seed_guard(g2, answers)
    assert g2.calls == 5 and g2.input_tokens == 5000
    c2 = FakeClient()
    s2 = mr.run_live(plan, candles, c2, answers, g2, window=300, equity=1000.0, min_interval=0.0,
                     sleep=lambda _s: None, log=lambda _m: None)
    assert c2.calls == 17 and s2["stop_reason"] == "plan complete" and s2["total_calls"] == 22
    done, recs = mr.load_done(answers)
    assert len(done) == 22 == len(recs)
    assert {(p[1], p[0]) for p in plan} == done
    first = recs[0]
    assert first["usage"] == {"input_tokens": 1000, "output_tokens": 12} and first["judgment"]["action"] == "hold"
    assert [r["bar_ms"] for r in recs] == sorted(r["bar_ms"] for r in recs)  # bar-major order


def test_projection_guard_stops_live_run(tmp_path: Path) -> None:
    candles, plan = _setup()
    g = _guard(1_000_000, check_after=3)  # 1000 tok/call over 1M planned calls = $42
    c = FakeClient()
    s = mr.run_live(plan, candles, c, tmp_path / "a.jsonl", g, window=300, equity=1000.0, min_interval=0.0,
                    sleep=lambda _s: None, log=lambda _m: None)
    assert c.calls == 3 and "projection" in s["stop_reason"]


def test_errors_stop_instead_of_looping(tmp_path: Path) -> None:
    candles, plan = _setup()
    c = FakeClient(fail=ConnectionError("down"))
    s = mr.run_live(plan, candles, c, tmp_path / "a.jsonl", _guard(len(plan)), window=300, equity=1000.0,
                    min_interval=0.0, sleep=lambda _s: None, log=lambda _m: None)
    assert c.calls == 5 * 3 and "in a row" in s["stop_reason"] and s["total_calls"] == 0

    class TypeSafeAuthenticationError(Exception):
        status = 401

    c2 = FakeClient(fail=TypeSafeAuthenticationError("bad key"))
    s2 = mr.run_live(plan, candles, c2, tmp_path / "b.jsonl", _guard(len(plan)), window=300, equity=1000.0,
                     min_interval=0.0, sleep=lambda _s: None, log=lambda _m: None)
    assert c2.calls == 1 and "non-retryable" in s2["stop_reason"]


def test_billed_unparsable_answer_is_charged_not_retried(tmp_path: Path) -> None:
    candles, plan = _setup()
    c = FakeClient(fail=mr.BilledParseError({"input_tokens": 800, "output_tokens": 5}, ValueError("x")))
    g = _guard(len(plan))
    s = mr.run_live(plan, candles, c, tmp_path / "a.jsonl", g, window=300, equity=1000.0, min_interval=0.0,
                    sleep=lambda _s: None, log=lambda _m: None)
    assert c.calls == 5 and g.calls == 5 and g.input_tokens == 4000 and "in a row" in s["stop_reason"]
    g2 = _guard(len(plan))
    mr.seed_guard(g2, tmp_path / "a.jsonl")
    assert g2.calls == 5 and g2.input_tokens == 4000
    lines = (tmp_path / "a.errors.jsonl").read_text().splitlines()
    assert all(json.loads(x)["billed"] for x in lines)
