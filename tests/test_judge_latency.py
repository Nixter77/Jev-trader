from __future__ import annotations

import time

from jev_trader.cycle import decision_payload, run_once
from jev_trader.execution import PaperBroker
from jev_trader.jev import judgment_from_dict


class _SlowJudge:
    model = "fake"

    def judge(self, _compact):
        time.sleep(0.02)
        return judgment_from_dict(
            {
                "action": "hold",
                "trend_aligned": 0.5,
                "false_break_risk": 0.5,
                "signal_strength": "нет края",
                "should_trade_now": 0.1,
                "model": "fake",
            }
        )

    def close(self) -> None:
        return None


def test_run_once_records_judge_ms(market_snapshot) -> None:
    result = run_once(market_snapshot, jev_client=_SlowJudge(), broker=PaperBroker())
    assert result.judge_ms is not None
    assert result.judge_ms >= 15.0
    payload = decision_payload(result)
    assert "judge_ms" in payload
    assert payload["judge_ms"] >= 15.0


def test_precomputed_judgment_has_no_judge_ms(market_snapshot, passing_answers) -> None:
    result = run_once(
        market_snapshot,
        judgment=judgment_from_dict(passing_answers),
        broker=PaperBroker(),
    )
    assert result.judge_ms is None
    assert "judge_ms" not in decision_payload(result)
