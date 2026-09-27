from __future__ import annotations

import pytest

from jev_trader.execution import ExchangeHTTPError, collect_day_income


def test_collect_day_income_sums_price_fee_and_funding() -> None:
    page = (
        200,
        [
            {
                "tranId": 1,
                "time": 10,
                "incomeType": "REALIZED_PNL",
                "symbol": "BTCUSDT",
                "income": "1.5",
            },
            {
                "tranId": 2,
                "time": 11,
                "incomeType": "COMMISSION",
                "symbol": "BTCUSDT",
                "income": "-0.2",
            },
            {
                "tranId": 3,
                "time": 12,
                "incomeType": "FUNDING_FEE",
                "symbol": "BTCUSDT",
                "income": "0.1",
            },
            {
                "tranId": 4,
                "time": 13,
                "incomeType": "TRANSFER",
                "symbol": "",
                "income": "100",
            },
        ],
    )

    def fetch(_params):
        return page

    out = collect_day_income(fetch, 0, 100)
    assert out is not None
    assert out["known"] is True
    assert out["truncated"] is False
    assert out["realized_usdt"] == pytest.approx(1.5)
    assert out["commission_usdt"] == pytest.approx(-0.2)
    assert out["funding_usdt"] == pytest.approx(0.1)
    assert out["net_usdt"] == pytest.approx(1.4)


def test_collect_day_income_is_unknown_when_the_window_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jev_trader.execution.DAY_INCOME_PAGE", 1)
    monkeypatch.setattr("jev_trader.execution.DAY_INCOME_MAX_PAGES", 2)

    def fetch(params):
        stamp = int(params["startTime"])
        return (
            200,
            [
                {
                    "tranId": stamp,
                    "time": stamp,
                    "incomeType": "COMMISSION",
                    "symbol": "XRPUSDT",
                    "income": "-1",
                }
            ],
        )

    out = collect_day_income(fetch, 1, 100)
    assert out is not None
    assert out["known"] is False
    assert out["truncated"] is True


def test_collect_day_income_backoff_raises_and_other_errors_are_unknown() -> None:
    def limited(_params):
        return 429, {"msg": "limit"}

    with pytest.raises(ExchangeHTTPError):
        collect_day_income(limited, 0, 1)

    def rejected(_params):
        return 400, {"code": -1}

    assert collect_day_income(rejected, 0, 1) is None
