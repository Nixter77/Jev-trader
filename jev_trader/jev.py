from __future__ import annotations

from typing import Any, Mapping

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

from jev_trader import JEV_MODEL
from jev_trader.models import (
    ACTIONS,
    SIGNAL_STRENGTH_LEVELS,
    CompactState,
    JevJudgment,
)

ACTION_CRITERIA: dict[str, str] = {
    "buy_long": "Преимущество у роста, вход в лонг оправдан.",
    "sell_short": "Преимущество у снижения, вход в шорт оправдан.",
    "close": "Открытую позицию лучше закрыть.",
    "hold": "Края нет, ждать.",
}

ACTION_INSTRUCTIONS = (
    "Выбрать действие на горизонте 6–12 баров рабочего ТФ с учётом тренда "
    "старшего ТФ, импульса, структуры и стакана."
)

FORBIDDEN_QUESTION_TERMS = (
    "размер",
    "size",
    "стоп",
    "stop",
    "плечо",
    "leverage",
    "целев",
    "target price",
    "почему",
    "rationale",
)


def build_questions() -> dict[str, Choice | Noul | Score]:
    """Five v1 questions in one system_one call. No size/stop/leverage/rationale."""
    return {
        "action": Choice(
            instructions=ACTION_INSTRUCTIONS,
            criteria=dict(ACTION_CRITERIA),
        ),
        "trend_aligned": Noul(
            instructions="Движение согласовано с режимом вышестоящего ТФ?",
        ),
        "false_break_risk": Noul(
            instructions="Высокий риск ложного пробоя текущего уровня / свинга?",
        ),
        "signal_strength": Score(
            instructions="Сила сигнала / края на горизонте 6–12 баров рабочего ТФ.",
            criteria=list(SIGNAL_STRENGTH_LEVELS),
        ),
        "should_trade_now": Noul(
            instructions="Сейчас стоит открывать, а не ждать следующий бар?",
        ),
    }


def state_for_jev(compact: CompactState) -> dict[str, Any]:
    payload = compact.as_dict()
    payload["summary"] = compact.as_text()
    return payload


def build_system_one_payload(compact: CompactState) -> dict[str, Any]:
    """Objects handed to TypeSafeClient.system_one (model pinned, five questions)."""
    return {
        "model": JEV_MODEL,
        "state": state_for_jev(compact),
        "questions": build_questions(),
    }


def _question_dump(question: Choice | Noul | Score) -> dict[str, Any]:
    if hasattr(question, "model_dump"):
        return question.model_dump()
    return dict(question)  # type: ignore[arg-type]


def payload_as_inspectable(compact: CompactState) -> dict[str, Any]:
    built = build_system_one_payload(compact)
    return {
        "model": built["model"],
        "state": built["state"],
        "questions": {name: _question_dump(q) for name, q in built["questions"].items()},
    }


def coerce_signal_level(raw: Any, score_value: float | None = None) -> str:
    if isinstance(raw, str):
        for level in SIGNAL_STRENGTH_LEVELS:
            if raw == level or raw.startswith(level):
                return level
        lowered = raw.lower()
        for level in SIGNAL_STRENGTH_LEVELS:
            if level.lower() in lowered:
                return level
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        idx = int(round(float(raw)))
        idx = max(0, min(idx, len(SIGNAL_STRENGTH_LEVELS) - 1))
        return SIGNAL_STRENGTH_LEVELS[idx]
    if score_value is not None:
        idx = int(round(score_value))
        idx = max(0, min(idx, len(SIGNAL_STRENGTH_LEVELS) - 1))
        return SIGNAL_STRENGTH_LEVELS[idx]
    return SIGNAL_STRENGTH_LEVELS[0]


def _legend_level(legend: Mapping[Any, Any], key: Any) -> str | None:
    if key in legend:
        return coerce_signal_level(legend[key])
    try:
        ikey = int(key)
    except (TypeError, ValueError):
        return None
    if ikey in legend:
        return coerce_signal_level(legend[ikey])
    if str(ikey) in legend:
        return coerce_signal_level(legend[str(ikey)])
    return None


def level_from_score_answer(score_answer: Any) -> tuple[str, float | None]:
    expected = getattr(score_answer, "score", None)
    probs = getattr(score_answer, "probabilities", None) or {}
    legend = getattr(score_answer, "legend", None) or {}
    if probs:
        best_key = max(probs.items(), key=lambda item: item[1])[0]
        labeled = _legend_level(legend, best_key)
        if labeled:
            return labeled, expected
        return coerce_signal_level(best_key, expected), expected
    if legend and expected is not None:
        labeled = _legend_level(legend, int(round(expected)))
        if labeled:
            return labeled, expected
    return coerce_signal_level(expected, expected), expected


def judgment_from_response(response: Any) -> JevJudgment:
    """Parse a typesafe-sdk SystemOneResponse into the v1 judgment."""
    choices = getattr(response, "choices", {}) or {}
    nouls = getattr(response, "nouls", {}) or {}
    scores = getattr(response, "scores", {}) or {}
    action_ans = choices["action"]
    action = action_ans.choice
    if action not in ACTIONS:
        raise ValueError(f"unexpected action from Jev: {action!r}")
    level, expected = level_from_score_answer(scores["signal_strength"])
    model = getattr(response, "model", JEV_MODEL)
    raw = {
        "action": action,
        "action_probabilities": dict(getattr(action_ans, "probabilities", {}) or {}),
        "trend_aligned": float(nouls["trend_aligned"].noul),
        "false_break_risk": float(nouls["false_break_risk"].noul),
        "should_trade_now": float(nouls["should_trade_now"].noul),
        "signal_strength": level,
        "signal_strength_score": expected,
        "model": model,
    }
    return JevJudgment(
        action=action,
        trend_aligned=float(nouls["trend_aligned"].noul),
        false_break_risk=float(nouls["false_break_risk"].noul),
        signal_strength=level,
        should_trade_now=float(nouls["should_trade_now"].noul),
        action_probabilities=dict(getattr(action_ans, "probabilities", {}) or {}),
        signal_strength_score=expected,
        model=str(model),
        raw=raw,
    )


def judgment_from_dict(data: Mapping[str, Any]) -> JevJudgment:
    action = str(data["action"])
    if action not in ACTIONS:
        raise ValueError(f"unexpected action: {action!r}")
    return JevJudgment(
        action=action,
        trend_aligned=float(data["trend_aligned"]),
        false_break_risk=float(data["false_break_risk"]),
        signal_strength=coerce_signal_level(data["signal_strength"]),
        should_trade_now=float(data["should_trade_now"]),
        action_probabilities=dict(data.get("action_probabilities") or {}),
        signal_strength_score=data.get("signal_strength_score"),
        model=str(data.get("model") or JEV_MODEL),
        raw=dict(data),
    )


class JevClient:
    """Thin adapter around typesafe-sdk TypeSafeClient. Model is always jev-1.13.0."""

    model = JEV_MODEL

    def __init__(self, api_key: str, client: TypeSafeClient | None = None) -> None:
        if not api_key and client is None:
            raise ValueError("TYPESAFE_API_KEY is missing")
        self._owns_client = client is None
        self._client = client or TypeSafeClient(api_key=api_key, model=self.model)

    def judge(self, compact: CompactState) -> JevJudgment:
        payload = build_system_one_payload(compact)
        response = self._client.system_one(
            state=payload["state"],
            questions=payload["questions"],
            model=payload["model"],
        )
        return judgment_from_response(response)

    def close(self) -> None:
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
