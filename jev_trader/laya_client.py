"""Laya (convaiinnovations/laya) System-1 backend — drop-in judge next to JevClient."""

from __future__ import annotations

import os
from typing import Any, Mapping

from jev_trader.jev import (
    allowed_actions_for_position,
    build_questions,
    coerce_signal_level,
    position_side_from_compact,
    state_for_jev,
)
from jev_trader.models import ACTIONS, CompactState, JevJudgment

# Avoid TensorFlow import deadlocks when transformers probes TF (Laya docs).
os.environ.setdefault("USE_TF", "0")


def _patch_torch_compile_if_needed() -> None:
    """Intel Mac + torch 2.2 + Python 3.12: ModernBert import hits unsupported Dynamo.

    Make torch.compile a no-op before transformers.models.modernbert loads.
    """
    try:
        import torch
    except ImportError:
        return
    # Dynamo unsupported on py3.12 with older torch; also harmless no-op elsewhere.
    ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    import sys
    if sys.version_info >= (3, 12) and ver < (2, 5):
        def _noop_compile(fn=None, *args, **kwargs):  # type: ignore[no-untyped-def]
            if fn is None:
                return lambda f: f
            return fn
        torch.compile = _noop_compile  # type: ignore[assignment]



LAYA_REPO = "convaiinnovations/laya"
LAYA_CHECKPOINTS: dict[str, tuple[str, str | None]] = {
    "english": (LAYA_REPO, None),
    "multilingual": (LAYA_REPO, "multilingual"),
    "typed-decisions": (LAYA_REPO, "typed-decisions"),
}
DEFAULT_LAYA_CHECKPOINT = "typed-decisions"



def build_laya_questions(
    allowed_actions: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Same five v1 questions as Jev, in Laya's dict schema (choice / noul / score)."""
    out: dict[str, dict[str, Any]] = {}
    for name, question in build_questions(allowed_actions=allowed_actions).items():
        qtype = type(question).__name__.lower()  # Choice / Noul / Score
        if qtype == "choice":
            out[name] = {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": dict(question.criteria),
            }
        elif qtype == "score":
            out[name] = {
                "type": "score",
                "instructions": question.instructions,
                "criteria": list(question.criteria),
            }
        elif qtype == "noul":
            out[name] = {
                "type": "noul",
                "instructions": question.instructions,
            }
        else:
            raise TypeError(f"unsupported question type for Laya: {type(question)!r}")
    return out



def _noul_value(answer: Mapping[str, Any]) -> float:
    if "noul" in answer:
        return float(answer["noul"])
    for key in ("probability", "p"):
        if key in answer:
            return float(answer[key])
    raise KeyError(f"noul answer missing probability: {sorted(answer)}")


def _signal_from_score_answer(answer: Mapping[str, Any]) -> tuple[str, float | None]:
    score = answer.get("score")
    expected = float(score) if score is not None else None
    legend = answer.get("legend") or {}
    probs = answer.get("probabilities") or {}
    if probs:
        best_key = max(probs.items(), key=lambda item: item[1])[0]
        if best_key in legend:
            return coerce_signal_level(legend[best_key], expected), expected
        if str(best_key) in legend:
            return coerce_signal_level(legend[str(best_key)], expected), expected
        return coerce_signal_level(best_key, expected), expected
    if legend and expected is not None:
        rounded = int(round(expected))
        if rounded in legend:
            return coerce_signal_level(legend[rounded], expected), expected
        if str(rounded) in legend:
            return coerce_signal_level(legend[str(rounded)], expected), expected
    return coerce_signal_level(expected, expected), expected


def judgment_from_laya_result(
    result: Mapping[str, Any],
    *,
    model_label: str,
) -> JevJudgment:
    """Map a Laya predict()/system_one() result onto JevJudgment."""
    answers = result.get("answers") or result
    action_ans = answers["action"]
    action = str(action_ans.get("choice") or action_ans.get("action"))
    if action not in ACTIONS:
        raise ValueError(f"unexpected action from Laya: {action!r}")
    level, expected = _signal_from_score_answer(answers["signal_strength"])
    probs = dict(action_ans.get("probabilities") or {})
    trend = _noul_value(answers["trend_aligned"])
    false_break = _noul_value(answers["false_break_risk"])
    should = _noul_value(answers["should_trade_now"])
    routing = result.get("routing")
    raw = {
        "action": action,
        "action_probabilities": probs,
        "trend_aligned": trend,
        "false_break_risk": false_break,
        "should_trade_now": should,
        "signal_strength": level,
        "signal_strength_score": expected,
        "model": model_label,
        "laya": {
            "routing": routing,
            "answers": dict(answers),
        },
    }
    return JevJudgment(
        action=action,
        trend_aligned=trend,
        false_break_risk=false_break,
        signal_strength=level,
        should_trade_now=should,
        action_probabilities=probs,
        signal_strength_score=expected,
        model=model_label,
        raw=raw,
    )


class LayaClient:
    """Local Laya System-1 judge. Same surface as JevClient: judge / close / context manager."""

    def __init__(
        self,
        *,
        checkpoint: str = DEFAULT_LAYA_CHECKPOINT,
        device: str | None = None,
        agent: Any | None = None,
        preload: bool = False,
    ) -> None:
        self.checkpoint = (checkpoint or DEFAULT_LAYA_CHECKPOINT).strip().lower()
        self.device = device
        self._preload = preload
        self._agent = agent
        self._owns_agent = agent is None
        self._model_label = f"laya:{self.checkpoint}"

    @property
    def model(self) -> str:
        return self._model_label

    def _ensure_agent(self) -> Any:
        if self._agent is not None:
            return self._agent
        try:
            _patch_torch_compile_if_needed()
            import laya
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Laya backend requested but package 'laya' is not installed. "
                "Install with: pip install 'jev-trader[laya]'   (or: pip install laya)"
            ) from exc

        if self.checkpoint in {"router", "route", "auto"}:
            from laya import Router

            kwargs: dict[str, Any] = {"preload": self._preload}
            if self.device:
                kwargs["device"] = self.device
            self._agent = Router(**kwargs)
            self._model_label = "laya:router"
            return self._agent

        if self.checkpoint not in LAYA_CHECKPOINTS:
            allowed = ", ".join([*LAYA_CHECKPOINTS, "router"])
            raise ValueError(
                f"unknown LAYA_CHECKPOINT={self.checkpoint!r}; expected one of: {allowed}"
            )

        repo, subfolder = LAYA_CHECKPOINTS[self.checkpoint]
        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        if subfolder:
            kwargs["subfolder"] = subfolder
        self._agent = laya.load(repo, **kwargs)
        self._model_label = f"laya:{self.checkpoint}"
        return self._agent

    def judge(self, compact: CompactState) -> JevJudgment:
        agent = self._ensure_agent()
        state = state_for_jev(compact)
        allowed = allowed_actions_for_position(position_side_from_compact(compact))
        questions = build_laya_questions(allowed_actions=allowed)
        predict = getattr(agent, "predict", None) or agent.system_one
        result = predict(state, questions)
        label = self._model_label
        if isinstance(result, Mapping):
            routing = result.get("routing") or {}
            if isinstance(routing, Mapping) and routing.get("model"):
                label = f"laya:{routing.get('model')}"

        judgment = judgment_from_laya_result(result, model_label=label)
        allowed_set = set(allowed)
        if judgment.action not in allowed_set:
            # Model picked a label we did not offer (or stale); wait instead of closing flat.
            from jev_trader.models import JevJudgment as _JJ
            judgment = _JJ(
                action="hold",
                trend_aligned=judgment.trend_aligned,
                false_break_risk=judgment.false_break_risk,
                signal_strength=judgment.signal_strength,
                should_trade_now=judgment.should_trade_now,
                action_probabilities=judgment.action_probabilities,
                signal_strength_score=judgment.signal_strength_score,
                model=judgment.model,
                raw={**(judgment.raw or {}), "remapped_from": judgment.action, "allowed": sorted(allowed_set)},
            )
        return judgment


    def close(self) -> None:
        if not self._owns_agent:
            return
        agent = self._agent
        self._agent = None
        if agent is None:
            return
        for name in ("unload", "close"):
            fn = getattr(agent, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001 — best-effort free
                    pass

    def __enter__(self) -> LayaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
