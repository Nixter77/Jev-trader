from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

PRODUCTION_HOSTS = frozenset(
    {
        "fapi.binance.com",
        "api.binance.com",
        "fstream.binance.com",
        "dapi.binance.com",
        "api.binance.us",
    }
)
TESTNET_HOST = "testnet.binancefuture.com"
LIVE_CONFIRM_VALUE = "I_UNDERSTAND"
PUBLIC_FAPI_REST = "https://fapi.binance.com"


@dataclass(frozen=True)
class Settings:
    typesafe_api_key: str
    binance_api_key: str
    binance_api_secret: str
    binance_fapi_base: str
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_notify: bool
    jev_model: str = "jev-1.13.0"
    decision_backend: str = "jev"
    laya_checkpoint: str = "multilingual"
    laya_device: str = ""


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def map_typesafe_api_key(environ: dict[str, str] | None = None) -> str | None:
    """Map `.env` `typesafe_API_KEY` onto the SDK's `TYPESAFE_API_KEY`."""
    env = os.environ if environ is None else environ
    existing = (env.get("TYPESAFE_API_KEY") or "").strip()
    if existing:
        return existing
    alt = (env.get("typesafe_API_KEY") or "").strip()
    if alt:
        env["TYPESAFE_API_KEY"] = alt
        return alt
    return None


def fapi_host(base_url: str) -> str:
    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    return (parsed.hostname or "").lower()


def is_production_base(base_url: str) -> bool:
    host = fapi_host(base_url)
    if host in PRODUCTION_HOSTS:
        return True
    if host.endswith(".binance.com") and "testnet" not in host:
        return True
    return False


def is_testnet_base(base_url: str) -> bool:
    return fapi_host(base_url) == TESTNET_HOST


def assert_not_production(base_url: str) -> None:
    if is_production_base(base_url):
        raise RuntimeError(
            f"refusing Binance production/mainnet base URL: {fapi_host(base_url)}"
        )


def load_settings(
    env_file: str | Path | None = None,
    *,
    allow_production: bool = False,
) -> Settings:
    if env_file is None:
        candidate = Path.cwd() / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
    else:
        load_dotenv(env_file, override=True)

    map_typesafe_api_key()

    base = (os.environ.get("BINANCE_FAPI_BASE") or "").strip()
    if not base:
        base = f"https://{TESTNET_HOST}"
    if not allow_production:
        assert_not_production(base)

    notify_raw = os.environ.get("TELEGRAM_NOTIFY")
    return Settings(
        typesafe_api_key=(os.environ.get("TYPESAFE_API_KEY") or "").strip(),
        binance_api_key=(os.environ.get("BINANCE_API_KEY") or "").strip(),
        binance_api_secret=(os.environ.get("BINANCE_API_SECRET") or "").strip(),
        binance_fapi_base=base.rstrip("/"),
        telegram_bot_token=(os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip(),
        telegram_chat_id=(os.environ.get("TELEGRAM_CHAT_ID") or "").strip(),
        telegram_notify=_truthy(notify_raw),
        jev_model="jev-1.13.0",
        decision_backend=(os.environ.get("DECISION_BACKEND") or "jev").strip().lower() or "jev",
        laya_checkpoint=(os.environ.get("LAYA_CHECKPOINT") or "multilingual").strip() or "multilingual",
        laya_device=(os.environ.get("LAYA_DEVICE") or "").strip(),
    )
