"""A reduce-only MARKET close drops resting entry limits before it reads the live size."""
from __future__ import annotations

from typing import Any

from jev_trader.execution import BinanceTestnetBroker, PositionsUnknown
from jev_trader.models import market_close_intent

SYM = "BTCUSDT"
STOP = {"orderId": 7, "symbol": SYM, "type": "STOP_MARKET", "side": "SELL", "closePosition": True, "stopPrice": "90"}
ENTRY = {"orderId": 8, "symbol": SYM, "type": "LIMIT", "side": "BUY", "timeInForce": "GTX", "origQty": "0.5"}


class Venue:
    """Resting BUY limit that fills the moment the bot looks away (after a size read)."""

    def __init__(self, *, size: float = 1.0, positions_known: bool = True) -> None:
        self.size = size
        self.open = [dict(STOP), dict(ENTRY)]
        self.positions_known = positions_known
        self.calls: list[tuple[str, str, dict]] = []

    def live_position(self, symbol: str) -> dict[str, Any] | None:
        if not self.positions_known:
            raise PositionsUnknown("balance only")
        seen = self.size
        if any(r["orderId"] == 8 for r in self.open):
            # The entry fills right after the read: the close qty is now short.
            self.open = [r for r in self.open if r["orderId"] != 8]
            self.size += 0.5
        return {"symbol": symbol, "side": "LONG", "size": seen} if seen > 0 else None

    def request(self, method: str, path: str, params: dict | None = None, signed: bool = False, timeout: float = 10.0, **_kw: Any):
        params = dict(params or {})
        self.calls.append((method, path, params))
        if method == "GET" and path == "/fapi/v1/openOrders":
            return 200, [dict(r) for r in self.open]
        if method == "DELETE" and path == "/fapi/v1/order":
            self.open = [r for r in self.open if r["orderId"] != params.get("orderId")]
            return 200, {"status": "CANCELED"}
        if method == "DELETE" and path == "/fapi/v1/allOpenOrders":
            self.open = []
            return 200, {"code": 200}
        if method == "POST" and path == "/fapi/v1/order":
            qty = min(float(params["quantity"]), self.size)
            self.size -= qty
            return 200, {"orderId": 9, "status": "FILLED", "executedQty": str(qty), "avgPrice": "100"}
        return 200, {}


def _broker(venue: Venue) -> BinanceTestnetBroker:
    b = BinanceTestnetBroker("k", "s", "https://testnet.binancefuture.com")
    b._time_synced = True
    b._fill_poll_attempts = 1
    b._fill_poll_sleep = 0
    b.filters_for = lambda symbol: {"stepSize": "0.001"}  # type: ignore[method-assign]
    b._live_position = venue.live_position  # type: ignore[method-assign]
    b._request = venue.request  # type: ignore[method-assign]
    return b


def _stop_alive(venue: Venue) -> bool:
    return any(r["orderId"] == 7 for r in venue.open)


def test_entry_filling_during_close_is_not_left_without_stop() -> None:
    venue = Venue()
    result = _broker(venue).submit(market_close_intent(symbol=SYM, qty=1.0, order_side="SELL", client_order_id="jc_1", risk_event="stop"))
    assert result.status == "filled"
    # Either the book is flat, or whatever is left still has its stop.
    assert venue.size == 0 or _stop_alive(venue)
    assert venue.size == 0


def test_blind_close_drops_resting_entry_but_keeps_stop() -> None:
    venue = Venue(positions_known=False)
    result = _broker(venue).submit(market_close_intent(symbol=SYM, qty=1.0, order_side="SELL", client_order_id="jc_2", risk_event="kill_switch"))
    assert result.status == "filled"
    assert not any(r["orderId"] == 8 for r in venue.open)  # no entry can re-open the long
    assert _stop_alive(venue)
