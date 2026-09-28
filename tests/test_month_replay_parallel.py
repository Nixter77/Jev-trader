"""jev_month_replay live path with several workers: fake client with latency and random 429s (no network)."""
from __future__ import annotations

import json
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import jev_month_replay as mr  # noqa: E402
from jev_trader.snapshot import make_uptrend_candles  # noqa: E402

RAW = {"action": "hold", "trend_aligned": 0.5, "false_break_risk": 0.5, "should_trade_now": 0.3,
       "signal_strength": "слабый", "model": "jev-1.13.0"}


class TypeSafeRateLimitError(Exception):
    status = 429

    def __init__(self, retry_after_ms: float | None) -> None:
        super().__init__("429")
        self.retry_after_ms = retry_after_ms


class TypeSafeAuthenticationError(Exception):
    status = 401


class FakeJev:
    """Thread-safe fake: random latency, random 429s; records every request and every answer per bar."""

    def __init__(self, p429: float = 0.0, seed: int = 0, latency: float = 0.003, input_tokens: int = 1000,
                 fail_after: int | None = None, fail: Exception | None = None, billed_every: int = 0) -> None:
        self.lock = threading.Lock()
        self.rng = random.Random(seed)
        self.p429, self.latency, self.input_tokens = p429, latency, input_tokens
        self.fail_after, self.fail, self.billed_every = fail_after, fail, billed_every
        self.requests: Counter = Counter()
        self.answers: Counter = Counter()
        self.limited: Counter = Counter()
        self.billed: Counter = Counter()
        self.after_fail = 0
        self.failed = False
        self.failing = False
        self.inflight = self.max_inflight = 0

    def judge(self, compact):  # noqa: ANN001
        key = compact.text
        with self.lock:
            self.requests[key] += 1
            if self.failed:
                self.after_fail += 1
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            lat = self.rng.random() * self.latency
            limited = self.rng.random() < self.p429
            fail_now = (self.fail_after is not None and not self.failed and not self.failing
                        and sum(self.answers.values()) >= self.fail_after)
            if fail_now:
                self.failing = True
            billed = (self.billed_every and not limited and not fail_now
                      and (sum(self.answers.values()) + sum(self.billed.values())) % self.billed_every == 0)
        try:
            time.sleep(lat)
            if fail_now:
                with self.lock:
                    self.failed = True  # requests counted from here on started after the error was returned
                raise self.fail
            if limited:
                with self.lock:
                    self.limited[key] += 1
                raise TypeSafeRateLimitError(self.rng.choice([None, 1.0, 3.0]))
            if billed:
                with self.lock:
                    self.billed[key] += 1
                raise mr.BilledParseError({"input_tokens": self.input_tokens, "output_tokens": 3}, ValueError("x"))
            with self.lock:
                self.answers[key] += 1
            return dict(RAW), {"input_tokens": self.input_tokens, "output_tokens": 12}, "jev-1.13.0"
        finally:
            with self.lock:
                self.inflight -= 1


def _setup(n: int = 110):
    candles = {s: make_uptrend_candles(n=n, seed=k) for k, s in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT"), 1)}
    start = candles["AAAUSDT"][0].ts
    end = candles["AAAUSDT"][-1].ts + mr.BAR_MS
    return candles, mr.build_plan(candles, start, end)


def _guard(plan_len: int, **kw) -> mr.BudgetGuard:
    base = dict(budget_usd=4.0, price_per_mtok=0.042, planned_calls=plan_len, max_calls=10_000, check_after=2000)
    base.update(kw)
    return mr.BudgetGuard(**base)


def _run(plan, candles, client, answers, guard, workers=6, **kw):
    opts = dict(window=300, equity=1000.0, workers=workers, min_interval=0.0, backoff_base=0.0,
                rate_limit_base=0.001, log=lambda _m: None)
    opts.update(kw)
    return mr.run_live(plan, candles, client, answers, guard, **opts)


def _lines(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    return [json.loads(x) for x in text.splitlines() if x]  # every line must parse: no interleaving


def test_parallel_with_429s_asks_each_bar_once(tmp_path: Path) -> None:
    candles, plan = _setup()
    assert len(plan) == 3 * 61
    fake = FakeJev(p429=0.3, seed=1)
    answers = tmp_path / "a.jsonl"
    g = _guard(len(plan))
    s = _run(plan, candles, fake, answers, g, rate_limit_attempts=50)
    assert s["stop_reason"] == "plan complete"
    assert fake.max_inflight > 1  # really concurrent
    assert sum(fake.limited.values()) > 0 and s["rate_limited"] == sum(fake.limited.values())
    assert set(fake.answers.values()) == {1} and len(fake.answers) == len(plan)  # no bar answered twice
    for key, n in fake.requests.items():  # a bar is only re-sent after a 429 for that bar
        assert n == 1 + fake.limited[key]
    recs = _lines(answers)
    assert len(recs) == len(plan) == g.calls
    assert {(r["symbol"], r["bar_ms"]) for r in recs} == {(p[1], p[0]) for p in plan}
    errs = _lines(answers.with_suffix(".errors.jsonl"))
    assert len(errs) == sum(fake.limited.values()) and all(e["rate_limited"] and not e["billed"] for e in errs)
    assert g.inflight == 0 and g.inflight_tokens == 0


def test_max_calls_never_exceeded(tmp_path: Path) -> None:
    candles, plan = _setup()
    for seed, cap in ((2, 37), (3, 1), (4, 64)):
        fake = FakeJev(p429=0.2, seed=seed)
        g = _guard(len(plan), max_calls=cap)
        s = _run(plan, candles, fake, tmp_path / f"m{seed}.jsonl", g, workers=8, rate_limit_attempts=50)
        assert sum(fake.answers.values()) == cap == g.calls and "max_calls" in s["stop_reason"]
        assert len(_lines(tmp_path / f"m{seed}.jsonl")) == cap


def test_budget_stop_respected_with_workers(tmp_path: Path) -> None:
    candles, plan = _setup()
    budget = 20 * 1000 / 1e6 * 0.042  # 20 calls of 1000 input tokens
    fake = FakeJev(p429=0.1, seed=5)
    g = _guard(len(plan), budget_usd=budget)
    s = _run(plan, candles, fake, tmp_path / "b.jsonl", g, workers=6, rate_limit_attempts=50)
    assert "budget" in s["stop_reason"]
    assert g.cost() <= budget + 1e-12
    assert 10 <= g.calls <= 20 and len(_lines(tmp_path / "b.jsonl")) == g.calls


def test_projection_check_counts_completed_calls(tmp_path: Path) -> None:
    candles, plan = _setup()
    fake = FakeJev(seed=6)
    g = _guard(1_000_000, check_after=40)  # $0.000042/call x 1M = $42 projected
    s = _run(plan, candles, fake, tmp_path / "p.jsonl", g, workers=6)
    assert "projection" in s["stop_reason"]
    assert 40 <= g.calls <= 40 + 5  # at most workers-1 in-flight calls complete after the check fires


def test_resume_after_interruption_skips_answered(tmp_path: Path) -> None:
    candles, plan = _setup()
    answers = tmp_path / "r.jsonl"
    first = FakeJev(p429=0.2, seed=7, fail_after=50, fail=TypeSafeAuthenticationError("bad key"))
    s1 = _run(plan, candles, first, answers, _guard(len(plan)), workers=5, rate_limit_attempts=50)
    assert "non-retryable" in s1["stop_reason"]
    assert first.after_fail <= 4  # auth error stops everything: only calls already in flight
    with answers.open("a") as fh:
        fh.write('{"symbol": "AAAUSDT", "bar_ms": 1, "judg')  # torn line from a crash
    done1 = set(mr.load_done(answers)[0])
    assert 50 <= len(done1) < len(plan)
    g2 = _guard(len(plan))
    mr.seed_guard(g2, answers)
    assert g2.calls == len(done1)
    second = FakeJev(p429=0.2, seed=8)
    s2 = _run(plan, candles, second, answers, g2, workers=5, rate_limit_attempts=50)
    assert s2["stop_reason"] == "plan complete"
    assert not (set(first.answers) & set(second.requests))  # nothing answered before is asked again
    assert sum(second.answers.values()) == len(plan) - len(done1)
    good = [json.loads(x) for x in answers.read_text().splitlines() if x.startswith("{") and x.endswith("}")]
    keys = [(r["symbol"], r["bar_ms"]) for r in good]
    assert len(keys) == len(set(keys)) == len(plan)  # no duplicates across both runs


def test_billed_unparsable_charged_once_and_skipped_on_resume(tmp_path: Path) -> None:
    candles, plan = _setup()
    answers = tmp_path / "u.jsonl"
    fake = FakeJev(seed=9, billed_every=10)
    g = _guard(len(plan))
    s = _run(plan, candles, fake, answers, g, workers=4, max_consecutive_errors=1000)
    assert s["stop_reason"] == "plan complete"
    nb = sum(fake.billed.values())
    assert nb > 0 and all(fake.requests[k] == 1 for k in fake.billed)  # never retried
    assert g.calls == len(plan) and g.input_tokens == 1000 * len(plan)  # billed ones are charged too
    g2 = _guard(len(plan))
    mr.seed_guard(g2, answers)
    assert g2.calls == len(plan)
    again = FakeJev(seed=10)
    s2 = _run(plan, candles, again, answers, g2, workers=4)
    assert sum(again.requests.values()) == 0 and s2["new_calls"] == 0  # resume does not re-pay them


def test_failure_streak_stops_all_workers(tmp_path: Path) -> None:
    candles, plan = _setup()
    class Down(FakeJev):
        def judge(self, compact):  # noqa: ANN001
            with self.lock:
                self.requests[compact.text] += 1
            raise ConnectionError("down")

    down = Down()
    s = _run(plan, candles, down, tmp_path / "f.jsonl", _guard(len(plan)), workers=4, attempts=3)
    assert "in a row" in s["stop_reason"] and s["total_calls"] == 0
    assert sum(down.requests.values()) <= (5 + 3) * 3


def test_rate_limit_pause_is_shared_and_respects_retry_after() -> None:
    clock = [100.0]
    p = mr.SharedPacer(min_interval=0.0, base=2.0, cap=300.0, clock=lambda: clock[0])
    assert p.rate_limited(7.5) == 7.5
    stop = threading.Event()
    got = []
    t = threading.Thread(target=lambda: got.append(p.wait_turn(stop)))
    t.start()
    time.sleep(0.05)
    assert t.is_alive()  # everyone waits for the shared pause
    clock[0] = 107.6
    t.join(timeout=2)
    assert got == [True]
    d = p.rate_limited(None)
    assert 4.0 <= d <= 5.0  # second strike without Retry-After: 2 x 2^1 with up to 25 % jitter
    assert mr.retry_after_seconds(TypeSafeRateLimitError(1500)) == 1.5

    class E(Exception):
        status = 418
        headers = {"retry-after": "3"}

    assert mr.is_rate_limited(E()) and mr.retry_after_seconds(E()) == 3.0


def test_env_file_missing_fails_before_anything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("typesafe_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="no such file"):
        mr.resolve_api_key(str(tmp_path / "nope.env"))
    empty = tmp_path / "e.env"
    empty.write_text("OTHER=1\n")
    with pytest.raises(SystemExit, match="nothing was called"):
        mr.resolve_api_key(str(empty))
    f = tmp_path / "k.env"
    f.write_text('typesafe_API_KEY="ts_fake_value"\nBINANCE_API_SECRET=zzz\n')
    import os
    before = dict(os.environ)
    key, src = mr.resolve_api_key(str(f))
    assert key == "ts_fake_value" and "k.env" in src and "ts_fake" not in src
    changed = dict(os.environ) != before  # boolean only: never let pytest print the environment
    assert not changed  # nothing from the file is exported


def test_progress_reports_spend(tmp_path: Path) -> None:
    candles, plan = _setup()
    answers = tmp_path / "live_answers.jsonl"
    answers.with_suffix(".meta.json").write_text(json.dumps({"planned_calls": len(plan), "budget_usd": 4.0}))
    fake = FakeJev(p429=0.2, seed=12, billed_every=25)
    _run(plan, candles, fake, answers, _guard(len(plan), max_calls=40), workers=4, rate_limit_attempts=50,
         max_consecutive_errors=1000)
    r = mr.progress(answers, 0.042)
    assert r["answers"] + r["billed_unparsable"] == 40 and r["input_tokens"] == 40_000
    assert abs(r["spent_usd"] - 40_000 / 1e6 * 0.042) < 1e-4 and r["planned_calls"] == len(plan)
    assert r["rate_limited"] == sum(fake.limited.values())
