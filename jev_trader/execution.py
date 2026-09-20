from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal, ROUND_DOWN
from typing import Any

from jev_trader.config import (
    LIVE_CONFIRM_VALUE,
    assert_not_production,
    is_production_base,
    is_testnet_base,
)
from jev_trader.http import ssl_context
from jev_trader.models import ExecutionResult, TradeIntent

SELL_ONLY_CLOSES_LONG = "sell_only_closes_long"
RECV_WINDOW_MS = 60_000


def format_binance_decimal(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def round_to_step(value: float, step: str | float) -> float:
    v = Decimal(str(value))
    s = Decimal(str(step))
    if s <= 0:
        return float(v)
    q = (v / s).to_integral_value(rounding=ROUND_DOWN) * s
    return float(q)


def parse_symbol_filters(info: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for sym in info.get("symbols") or []:
        if not isinstance(sym, dict):
            continue
        name = str(sym.get("symbol") or "").upper()
        if not name:
            continue
        packed: dict[str, str] = {}
        for flt in sym.get("filters") or []:
            if not isinstance(flt, dict):
                continue
            kind = flt.get("filterType")
            if kind == "LOT_SIZE":
                packed["stepSize"] = str(flt.get("stepSize") or "0")
                packed["minQty"] = str(flt.get("minQty") or "0")
            elif kind == "PRICE_FILTER":
                packed["tickSize"] = str(flt.get("tickSize") or "0")
            elif kind in {"MIN_NOTIONAL", "NOTIONAL"}:
                packed["minNotional"] = str(flt.get("notional") or flt.get("minNotional") or "0")
        if packed:
            out[name] = packed
    return out


def parse_usdt_wallet(payload: Any) -> dict[str, Any]:
    """USDT wallet + non-zero UM positions from /fapi/v2/balance or /fapi/v2/account."""
    equity = 0.0
    available = 0.0
    positions: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            if str(row.get("asset") or "").upper() == "USDT":
                equity = float(row.get("balance") or row.get("walletBalance") or 0.0)
                available = float(row.get("availableBalance") or equity)
            amt = float(row.get("positionAmt") or 0.0)
            if abs(amt) > 0:
                positions.append(_position_row(row, amt))
    elif isinstance(payload, dict):
        equity = float(payload.get("totalWalletBalance") or payload.get("availableBalance") or 0.0)
        available = float(payload.get("availableBalance") or equity)
        for row in payload.get("assets") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("asset") or "").upper() == "USDT":
                equity = float(row.get("walletBalance") or row.get("marginBalance") or equity)
                available = float(row.get("availableBalance") or available or equity)
        for row in payload.get("positions") or []:
            if not isinstance(row, dict):
                continue
            amt = float(row.get("positionAmt") or 0.0)
            if abs(amt) > 0:
                positions.append(_position_row(row, amt))
    else:
        raise ValueError("wallet payload must be a list or object")
    return {
        "equity_usdt": equity,
        "available_usdt": available,
        "open_positions": len(positions),
        "positions": positions,
    }


def _position_row(row: dict[str, Any], amt: float) -> dict[str, Any]:
    side = "LONG" if amt > 0 else "SHORT"
    return {
        "symbol": str(row.get("symbol") or "").upper(),
        "side": side,
        "size": abs(amt),
        "entry": float(row.get("entryPrice") or 0.0) or None,
        "unrealized_pnl_usdt": float(row.get("unrealizedProfit") or 0.0),
    }


def open_sell_error(intent: TradeIntent) -> str | None:
    """SELL without reduceOnly would open a short. That is not allowed."""
    if intent.skip_reason or intent.qty <= 0:
        return None
    if intent.order_side == "SELL" and not intent.reduce_only:
        return SELL_ONLY_CLOSES_LONG
    if intent.action == "sell_short" and not intent.reduce_only:
        return SELL_ONLY_CLOSES_LONG
    return None


class PaperBroker:
    venue = "paper"

    def submit(self, intent: TradeIntent) -> ExecutionResult:
        if intent.skip_reason or intent.qty <= 0:
            return ExecutionResult(
                status="skipped",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=intent.reduce_only,
                detail={"skip_reason": intent.skip_reason},
            )
        blocked = open_sell_error(intent)
        if blocked:
            return ExecutionResult(
                status="rejected",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=intent.reduce_only,
                detail={"error": blocked},
            )
        return ExecutionResult(
            status="paper_recorded",
            venue=self.venue,
            client_order_id=intent.client_order_id,
            reduce_only=intent.reduce_only,
            detail={
                "action": intent.action,
                "qty": intent.qty,
                "entry_type": intent.entry_type,
                "stop_price": intent.stop_price,
                "limit_price": intent.limit_price,
                "reduce_only": intent.reduce_only,
            },
        )


class BinanceFuturesBroker:
    """Signed USD-M orders. Testnet by default; production only with live lock."""

    venue = "binance_testnet"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        *,
        live: bool = False,
        allow_live: bool = False,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.base_url = base_url.rstrip("/")
        if live:
            confirmed = allow_live or (
                os.environ.get("BINANCE_ALLOW_LIVE", "").strip() == LIVE_CONFIRM_VALUE
            )
            if not confirmed:
                raise RuntimeError(
                    "refusing Binance production/mainnet: set BINANCE_ALLOW_LIVE=I_UNDERSTAND"
                )
            if not is_production_base(self.base_url):
                raise RuntimeError(
                    f"live venue requires production UM host fapi.binance.com, got {self.base_url}"
                )
            self.venue = "binance_live"
        else:
            assert_not_production(self.base_url)
            if not is_testnet_base(self.base_url):
                raise RuntimeError(
                    f"only Binance USD-M futures testnet is allowed: {self.base_url}"
                )
            self.venue = "binance_testnet"
        self._time_offset_ms = 0
        self._time_synced = False
        self._wallet_cache: dict[str, Any] | None = None
        self._wallet_ts = 0.0
        self._filters: dict[str, dict[str, str]] = {}
        self._filters_loaded = False

    def sync_time(self) -> int:
        status, data = self._request("GET", "/fapi/v1/time", signed=False)
        if status == 200 and isinstance(data, dict) and data.get("serverTime"):
            server = int(data["serverTime"])
            self._time_offset_ms = server - int(time.time() * 1000)
            self._time_synced = True
            return server
        self._time_synced = True
        return int(time.time() * 1000) + self._time_offset_ms

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        timeout: float = 10.0,
        *,
        _retried: bool = False,
    ) -> tuple[int, Any]:
        params = dict(params or {})
        if signed:
            params.setdefault("recvWindow", RECV_WINDOW_MS)
            params["timestamp"] = self._timestamp_ms()
            query = urllib.parse.urlencode(params)
            signature = hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()
            query = f"{query}&signature={signature}"
        else:
            query = urllib.parse.urlencode(params)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        headers = {"X-MBX-APIKEY": self.api_key, "Accept": "application/json"}
        req = urllib.request.Request(url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body) if body else {}
                status = resp.status
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body) if body else {"error": str(exc)}
            except json.JSONDecodeError:
                data = {"error": body or str(exc)}
            status = exc.code
        except Exception as exc:  # noqa: BLE001 — ping/order must surface transport errors
            return 0, {"error": f"{type(exc).__name__}: {exc}"}
        if (
            signed
            and not _retried
            and isinstance(data, dict)
            and data.get("code") == -1021
        ):
            self.sync_time()
            params.pop("timestamp", None)
            return self._request(
                method, path, params=params, signed=True, timeout=timeout, _retried=True
            )
        return status, data

    def ping(self) -> dict[str, Any]:
        status, data = self._request("GET", "/fapi/v1/ping")
        return {"http_status": status, "body": data, "base": self.base_url, "venue": self.venue}

    def fetch_wallet(self, *, ttl: float = 10.0) -> dict[str, Any]:
        now = time.monotonic()
        if self._wallet_cache is not None and now - self._wallet_ts < ttl:
            return self._wallet_cache
        if not self._time_synced:
            self.sync_time()
        status, data = self._request("GET", "/fapi/v2/account", signed=True)
        if not (200 <= status < 300) or not isinstance(data, dict) or data.get("code"):
            status, data = self._request("GET", "/fapi/v2/balance", signed=True)
        if not (200 <= status < 300):
            raise RuntimeError(f"binance wallet HTTP {status}: {data}")
        wallet = parse_usdt_wallet(data)
        wallet["http_status"] = status
        wallet["venue"] = self.venue
        wallet["base"] = self.base_url
        self._wallet_cache = wallet
        self._wallet_ts = now
        return wallet

    def filters_for(self, symbol: str) -> dict[str, str]:
        if not self._filters_loaded:
            status, data = self._request("GET", "/fapi/v1/exchangeInfo")
            self._filters_loaded = True
            if 200 <= status < 300 and isinstance(data, dict):
                self._filters = parse_symbol_filters(data)
        return self._filters.get(symbol.upper(), {})

    def submit(self, intent: TradeIntent) -> ExecutionResult:
        if intent.skip_reason or intent.qty <= 0:
            return ExecutionResult(
                status="skipped",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=intent.reduce_only,
                detail={"skip_reason": intent.skip_reason},
            )
        blocked = open_sell_error(intent)
        if blocked:
            return ExecutionResult(
                status="rejected",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=intent.reduce_only,
                detail={"error": blocked},
            )
        qty = intent.qty
        price = intent.limit_price
        flt = self.filters_for(intent.symbol)
        if flt:
            if flt.get("stepSize"):
                qty = round_to_step(qty, flt["stepSize"])
            if price is not None and flt.get("tickSize"):
                price = round_to_step(price, flt["tickSize"])
            min_qty = float(flt.get("minQty") or 0.0)
            if qty <= 0 or (min_qty and qty < min_qty):
                return ExecutionResult(
                    status="rejected",
                    venue=self.venue,
                    client_order_id=intent.client_order_id,
                    reduce_only=intent.reduce_only,
                    detail={"error": "qty below LOT_SIZE", "qty": qty, "filters": flt},
                )
            min_notional = float(flt.get("minNotional") or 0.0)
            notion = qty * float(price or 0.0)
            if min_notional and price and notion < min_notional:
                return ExecutionResult(
                    status="rejected",
                    venue=self.venue,
                    client_order_id=intent.client_order_id,
                    reduce_only=intent.reduce_only,
                    detail={"error": "below minNotional", "notional": notion, "filters": flt},
                )
        params: dict[str, Any] = {
            "symbol": intent.symbol,
            "side": intent.order_side,
            "type": "MARKET" if intent.entry_type == "MARKET" else "LIMIT",
            "quantity": format_binance_decimal(qty),
            "newClientOrderId": intent.client_order_id,
            "reduceOnly": "true" if intent.reduce_only else "false",
        }
        if params["type"] == "LIMIT":
            if price is None or price <= 0:
                return ExecutionResult(
                    status="rejected",
                    venue=self.venue,
                    client_order_id=intent.client_order_id,
                    reduce_only=intent.reduce_only,
                    detail={"error": "LIMIT order requires limit_price"},
                )
            params["timeInForce"] = "GTX"  # post-only
            params["price"] = format_binance_decimal(price)
        status, data = self._request("POST", "/fapi/v1/order", params=params, signed=True)
        ok = 200 <= status < 300
        return ExecutionResult(
            status="accepted" if ok else "rejected",
            venue=self.venue,
            client_order_id=intent.client_order_id,
            reduce_only=intent.reduce_only,
            detail={"http_status": status, "body": data},
        )


class BinanceTestnetBroker(BinanceFuturesBroker):
    def __init__(self, api_key: str, api_secret: str, base_url: str) -> None:
        super().__init__(api_key, api_secret, base_url, live=False)
