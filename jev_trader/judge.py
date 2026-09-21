"""Decision-backend factory: Jev (TypeSafe API) or Laya (local HF weights)."""

from __future__ import annotations

from typing import Protocol

from jev_trader.config import Settings
from jev_trader.jev import JevClient
from jev_trader.laya_client import DEFAULT_LAYA_CHECKPOINT, LayaClient
from jev_trader.models import CompactState, JevJudgment

BACKEND_JEV = "jev"
BACKEND_LAYA = "laya"
VALID_BACKENDS = frozenset({BACKEND_JEV, BACKEND_LAYA})


class JudgeClient(Protocol):
    @property
    def model(self) -> str: ...

    def judge(self, compact: CompactState) -> JevJudgment: ...

    def close(self) -> None: ...


def normalize_backend(value: str | None) -> str:
    raw = (value or BACKEND_JEV).strip().lower()
    if raw in {"typesafe", "jev-1", "jev1"}:
        return BACKEND_JEV
    if raw in {"laya", "hf", "huggingface", "local"}:
        return BACKEND_LAYA
    if raw not in VALID_BACKENDS:
        raise ValueError(f"unknown decision backend {value!r}; expected 'jev' or 'laya'")
    return raw


def make_judge_client(settings: Settings) -> JudgeClient:
    """Build the active judge from Settings.decision_backend."""
    backend = normalize_backend(getattr(settings, "decision_backend", BACKEND_JEV))
    if backend == BACKEND_LAYA:
        return LayaClient(
            checkpoint=getattr(settings, "laya_checkpoint", None) or DEFAULT_LAYA_CHECKPOINT,
            device=(getattr(settings, "laya_device", None) or None) or None,
        )
    if not settings.typesafe_api_key:
        raise ValueError("TYPESAFE_API_KEY is missing (decision backend=jev)")
    return JevClient(api_key=settings.typesafe_api_key)
