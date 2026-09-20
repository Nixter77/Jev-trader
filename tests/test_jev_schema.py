from __future__ import annotations

from types import SimpleNamespace

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

from jev_trader import JEV_MODEL
from jev_trader.jev import (
    JevClient,
    build_questions,
    build_system_one_payload,
    judgment_from_response,
)
from jev_trader.state import build_compact_state


def test_system_one_payload_pins_model_and_five_questions(market_snapshot) -> None:
    compact = build_compact_state(market_snapshot)
    payload = build_system_one_payload(compact)
    assert payload["model"] == "jev-1.13.0"
    assert payload["model"] == JEV_MODEL
    assert payload["model"] != "jev-latest"

    questions = payload["questions"]
    assert list(questions) == [
        "action",
        "trend_aligned",
        "false_break_risk",
        "signal_strength",
        "should_trade_now",
    ]
    assert isinstance(questions["action"], Choice)
    assert isinstance(questions["trend_aligned"], Noul)
    assert isinstance(questions["false_break_risk"], Noul)
    assert isinstance(questions["should_trade_now"], Noul)
    assert isinstance(questions["signal_strength"], Score)

    assert set(questions["action"].criteria) == {
        "buy_long",
        "sell_short",
        "close",
        "hold",
    }
    assert list(questions["signal_strength"].criteria) == [
        "нет края",
        "слабый",
        "рабочий",
        "сильный",
    ]

    instructions = " ".join(str(q.instructions) for q in questions.values()).lower()
    for banned in (
        "size",
        "leverage",
        "rationale",
        "target price",
        "почему",
        "размер позиции",
        "стоп-лосс",
        "stop loss",
        "плечо",
    ):
        assert banned not in instructions


def test_jev_client_judge_sends_pinned_model_and_five_questions(market_snapshot) -> None:
    compact = build_compact_state(market_snapshot)
    fake_response = SimpleNamespace(
        model="jev-1.13.0",
        choices={
            "action": SimpleNamespace(
                choice="hold",
                probabilities={
                    "buy_long": 0.05,
                    "sell_short": 0.05,
                    "close": 0.05,
                    "hold": 0.85,
                },
            )
        },
        nouls={
            "trend_aligned": SimpleNamespace(noul=0.41),
            "false_break_risk": SimpleNamespace(noul=0.55),
            "should_trade_now": SimpleNamespace(noul=0.20),
        },
        scores={
            "signal_strength": SimpleNamespace(
                score=0.4,
                probabilities={0: 0.7, 1: 0.2, 2: 0.1, 3: 0.0},
                legend={0: "нет края", 1: "слабый", 2: "рабочий", 3: "сильный"},
            )
        },
    )

    class RecordingSDK:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def system_one(self, state, questions, model=None):
            self.calls.append({"state": state, "questions": questions, "model": model})
            return fake_response

    sdk = RecordingSDK()
    client = JevClient(api_key="unused", client=sdk)  # type: ignore[arg-type]
    judgment = client.judge(compact)
    assert len(sdk.calls) == 1
    call = sdk.calls[0]
    assert call["model"] == "jev-1.13.0"
    assert set(call["questions"]) == set(build_questions())
    assert "candles" not in call["state"]
    assert judgment.action == "hold"
    assert 0.0 <= judgment.trend_aligned <= 1.0
    assert judgment.signal_strength == "нет края"
    parsed = judgment_from_response(fake_response)
    assert parsed.action in {"buy_long", "sell_short", "close", "hold"}


def test_jev_client_wraps_typesafe_sdk_pinned_model() -> None:
    client = JevClient(api_key="ts_dummy_not_used")
    assert isinstance(client._client, TypeSafeClient)
    assert client.model == "jev-1.13.0"
    client.close()
