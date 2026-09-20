from __future__ import annotations

from pathlib import Path

from jev_trader.config import is_production_base, load_settings, map_typesafe_api_key


def test_maps_typesafe_lowercase_key(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    env = {"typesafe_API_KEY": "ts_test_key"}
    mapped = map_typesafe_api_key(env)
    assert mapped == "ts_test_key"
    assert env["TYPESAFE_API_KEY"] == "ts_test_key"


def test_load_settings_maps_and_refuses_to_log_secret(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "typesafe_API_KEY=ts_from_file",
                "BINANCE_API_KEY=bk",
                "BINANCE_API_SECRET=bs",
                "BINANCE_FAPI_BASE=https://testnet.binancefuture.com",
                "TELEGRAM_BOT_TOKEN=tg",
                "TELEGRAM_CHAT_ID=1",
                "TELEGRAM_NOTIFY=0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("typesafe_API_KEY", raising=False)
    settings = load_settings(env_file)
    assert settings.typesafe_api_key == "ts_from_file"
    assert settings.jev_model == "jev-1.13.0"
    assert not is_production_base(settings.binance_fapi_base)


def test_production_hosts_detected() -> None:
    assert is_production_base("https://fapi.binance.com")
    assert not is_production_base("https://testnet.binancefuture.com")
