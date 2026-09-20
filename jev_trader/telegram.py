from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from jev_trader.config import Settings
from jev_trader.http import ssl_context


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, enabled: bool) -> None:
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled and bool(token) and bool(chat_id)

    def send(self, text: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        body = json.dumps(
            {"chat_id": self.chat_id, "text": text[:3500], "disable_web_page_preview": True}
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10, context=ssl_context()) as resp:
                return {"http_status": resp.status, "ok": True}
        except urllib.error.HTTPError as exc:
            return {"http_status": exc.code, "ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def notifier_from_settings(settings: Settings, force_off: bool = False) -> TelegramNotifier:
    return TelegramNotifier(
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        enabled=settings.telegram_notify and not force_off,
    )
