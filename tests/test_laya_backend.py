from __future__ import annotations

from dataclasses import replace

import pytest

from jev_trader.config import Settings, load_settings
from jev_trader.judge import make_judge_client, normalize_backend
from jev_trader.laya_client import (
    LayaClient,
    build_laya_questions,
    judgment_from_laya_result,
)
from jev_trader.models import CompactState


def _settings(**kwargs) -> Settings:
    base = Settings(
        typesafe_api_key="ts_test",
        binance_api_key="",
        binance_api_secret="",
        binance_fapi_base="https://testnet.binancefuture.com",
        telegram_bot_token="",
        telegram_chat_id="",
        telegram_notify=False,
    )
    return replace(base, **kwargs)


def test_normalize_backend_aliases() -> None:
    assert normalize_backend("jev") == "jev"
    assert normalize_backend("typesafe") == "jev"
    assert normalize_backend("laya") == "laya"
    assert normalize_backend("hf") == "laya"
    with pytest.raises(ValueError):
        normalize_backend("gpt")


def test_build_laya_questions_mirrors_jev_schema() -> None:
    qs = build_laya_questions()
    assert list(qs) == [
        "action",
        "trend_aligned",
        "false_break_risk",
        "signal_strength",
        "should_trade_now",
    ]
    assert qs["action"]["type"] == "choice"
    assert set(qs["action"]["criteria"]) == {"buy_long", "sell_short", "close", "hold"}
    assert qs["trend_aligned"]["type"] == "noul"
    assert qs["signal_strength"]["type"] == "score"
    assert qs["should_trade_now"]["type"] == "noul"


def test_judgment_from_laya_result_maps_fields() -> None:
    result = {
        "answers": {
            "action": {
                "choice": "buy_long",
                "probabilities": {
                    "buy_long": 0.55,
                    "sell_short": 0.1,
                    "close": 0.05,
                    "hold": 0.3,
                },
            },
            "trend_aligned": {"noul": 0.81},
            "false_break_risk": {"noul": 0.22},
            "signal_strength": {
                "score": 2.1,
                "legend": {
                    "0": "нет края",
                    "1": "слабый",
                    "2": "рабочий",
                    "3": "сильный",
                },
                "probabilities": {"0": 0.05, "1": 0.1, "2": 0.6, "3": 0.25},
            },
            "should_trade_now": {"noul": 0.77},
        }
    }
    judgment = judgment_from_laya_result(result, model_label="laya:test")
    assert judgment.action == "buy_long"
    assert judgment.trend_aligned == pytest.approx(0.81)
    assert judgment.false_break_risk == pytest.approx(0.22)
    assert judgment.should_trade_now == pytest.approx(0.77)
    assert judgment.signal_strength == "рабочий"
    assert judgment.model == "laya:test"
    assert judgment.action_probabilities["buy_long"] == pytest.approx(0.55)


def test_laya_client_judge_with_fake_agent() -> None:
    class FakeAgent:
        def predict(self, state, questions):
            assert "action" in questions
            assert "summary" in state or isinstance(state, dict)
            return {
                "answers": {
                    "action": {
                        "choice": "hold",
                        "probabilities": {
                            "buy_long": 0.05,
                            "sell_short": 0.05,
                            "close": 0.1,
                            "hold": 0.8,
                        },
                    },
                    "trend_aligned": {"noul": 0.4},
                    "false_break_risk": {"noul": 0.5},
                    "signal_strength": {
                        "score": 0.4,
                        "legend": {
                            0: "нет края",
                            1: "слабый",
                            2: "рабочий",
                            3: "сильный",
                        },
                        "probabilities": {0: 0.7, 1: 0.2, 2: 0.1, 3: 0.0},
                    },
                    "should_trade_now": {"noul": 0.15},
                }
            }

    client = LayaClient(agent=FakeAgent())
    compact = CompactState(payload={"symbol": "BTCUSDT"}, text="BTCUSDT flat")
    judgment = client.judge(compact)
    assert judgment.action == "hold"
    assert judgment.signal_strength == "нет края"
    assert judgment.model.startswith("laya:")
    client.close()


def test_factory_selects_laya_without_typesafe_key() -> None:
    settings = _settings(typesafe_api_key="", decision_backend="laya", laya_checkpoint="multilingual")
    client = make_judge_client(settings)
    assert isinstance(client, LayaClient)
    assert client.checkpoint == "multilingual"


def test_factory_jev_requires_key() -> None:
    settings = _settings(typesafe_api_key="", decision_backend="jev")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        make_judge_client(settings)


def test_load_settings_reads_backend_env(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BINANCE_FAPI_BASE=https://testnet.binancefuture.com",
                "typesafe_API_KEY=ts_x",
                "DECISION_BACKEND=laya",
                "LAYA_CHECKPOINT=english",
                "LAYA_DEVICE=cpu",
            ]
        )
        + "\n"
    )
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("DECISION_BACKEND", raising=False)
    monkeypatch.delenv("LAYA_CHECKPOINT", raising=False)
    monkeypatch.delenv("LAYA_DEVICE", raising=False)
    settings = load_settings(env_file)
    assert settings.decision_backend == "laya"
    assert settings.laya_checkpoint == "english"
    assert settings.laya_device == "cpu"
