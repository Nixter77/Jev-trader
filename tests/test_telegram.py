from jev_trader.telegram import TelegramNotifier


def test_disabled_notifier_does_not_send() -> None:
    notifier = TelegramNotifier(token="x", chat_id="1", enabled=False)
    assert notifier.send("decision") is None
