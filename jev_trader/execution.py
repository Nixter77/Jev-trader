from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from dataclasses import replace
from typing import Any, Callable

from jev_trader.config import (
    LIVE_CONFIRM_VALUE,
    assert_not_production,
    is_production_base,
    is_testnet_base,
)
from jev_trader.http import ssl_context
from jev_trader.models import ExecutionResult, TradeIntent, market_close_intent
from jev_trader.status import log_event

SELL_ONLY_CLOSES_LONG = "sell_only_closes_long"
RECV_WINDOW_MS = 60_000
FILLED_ORDER_STATUSES = frozenset({"FILLED", "PARTIALLY_FILLED"})
LIMIT_ORDER_TYPES = frozenset({"LIMIT", "LIMIT_MAKER"})
CLOSE_STOP_TYPES = frozenset({"STOP", "STOP_MARKET"})
# Temporary reduce-only STOP_MARKET (explicit qty) that covers a closePosition
# stop swap. client_order_id(symbol, "stop_tmp") starts with this.
TEMP_STOP_PREFIX = "jt"
# Binance rejects newClientOrderId longer than 36 characters.
_CLIENT_ID_MAX = 36

order_lock = threading.RLock()
flatten_generation = 0
_id_lock = threading.Lock()
_id_seq = 0


def bump_flatten_generation() -> int:
    """Mark that a force-close started. In-flight entries must not reopen."""
    global flatten_generation
    with order_lock:
        flatten_generation += 1
        return flatten_generation


def current_flatten_generation() -> int:
    with order_lock:
        return flatten_generation


def client_order_id(symbol: str, action: str) -> str:
    """Short id. `jev1_1000SHIBUSDT_buy_long_<ms>` does not fit Binance's 36-char limit."""
    global _id_seq
    tag = {"buy_long": "b", "close": "c", "sell_short": "s", "flatten": "f", "stop_tmp": "t"}.get(action, "x")
    sym = "".join(ch for ch in symbol.upper() if ch.isalnum())[:12]
    with _id_lock:
        _id_seq = (_id_seq + 1) % 1000
        seq = _id_seq
    cid = f"j{tag}{sym}{int(time.time() * 1000)}{seq:03d}"
    return cid[:_CLIENT_ID_MAX]


def format_binance_decimal(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def round_to_step(value: float, step: str | float, *, rounding=ROUND_DOWN) -> float:
    v = Decimal(str(value))
    s = Decimal(str(step))
    if s <= 0:
        return float(v)
    q = (v / s).to_integral_value(rounding=rounding) * s
    return float(q)


def gtx_inside_price(price: float, order_side: str, tick: str | float) -> float:
    """One tick away from the touch so a post-only order does not cross and get -5022."""
    step = Decimal(str(tick))
    if step <= 0 or price <= 0:
        return price
    nudged = Decimal(str(price)) + (step if order_side == "SELL" else -step)
    rounding = ROUND_UP if order_side == "SELL" else ROUND_DOWN
    q = (nudged / step).to_integral_value(rounding=rounding) * step
    out = float(q)
    return out if out > 0 else price


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
        # Margin balance includes unrealized PnL. Wallet balance does not, so a
        # daily-loss limit keyed off it never sees an open loss.
        margin = payload.get("totalMarginBalance")
        equity = float(
            margin
            if margin not in (None, "")
            else payload.get("totalWalletBalance") or payload.get("availableBalance") or 0.0
        )
        available = float(payload.get("availableBalance") or equity)
        for row in payload.get("assets") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("asset") or "").upper() == "USDT":
                if margin in (None, ""):
                    equity = float(row.get("marginBalance") or row.get("walletBalance") or equity)
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



def close_side_for_position(position_side: str) -> str:
    """One-way mode: SELL closes LONG, BUY covers SHORT."""
    return "BUY" if position_side == "SHORT" else "SELL"


def is_reduce_only_rejected(body: Any) -> bool:
    return isinstance(body, dict) and body.get("code") == -2022


def blocked_execution(intent: TradeIntent, venue: str) -> ExecutionResult | None:
    """Skip or reject before any venue sees the order. None means the order may proceed."""
    if intent.skip_reason or intent.qty <= 0:
        return ExecutionResult(
            status="skipped",
            venue=venue,
            client_order_id=intent.client_order_id,
            reduce_only=intent.reduce_only,
            detail={"skip_reason": intent.skip_reason},
        )
    blocked = open_sell_error(intent)
    if blocked:
        return ExecutionResult(
            status="rejected",
            venue=venue,
            client_order_id=intent.client_order_id,
            reduce_only=intent.reduce_only,
            detail={"error": blocked},
        )
    return None


def _detail(execution: ExecutionResult) -> dict[str, Any]:
    return execution.detail if isinstance(execution.detail, dict) else {}


def _body(execution: ExecutionResult) -> Any:
    detail = execution.detail if isinstance(execution.detail, dict) else None
    if detail is None:
        return None
    return detail.get("body")


def order_executed_qty(body: Any) -> float:
    if not isinstance(body, dict):
        return 0.0
    try:
        return float(body.get("executedQty") or body.get("cumQty") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def order_fill_price(body: Any) -> float | None:
    if not isinstance(body, dict):
        return None
    try:
        avg = float(body.get("avgPrice") or 0.0)
    except (TypeError, ValueError):
        avg = 0.0
    if avg > 0:
        return avg
    try:
        px = float(body.get("price") or 0.0)
    except (TypeError, ValueError):
        return None
    return px if px > 0 else None


# -1007: "Timeout waiting for response from backend server. Send status unknown".
# -1006: unexpected response from the message bus; execution status unknown.
SEND_UNKNOWN_CODES = frozenset({-1006, -1007})


def wallet_positions_known(wallet: Any) -> bool:
    """False when the wallet came without a positions list (e.g. /fapi/v2/balance).

    Such a wallet still has a valid balance, but "no positions" in it means
    "unknown", not "flat". Wallets without the flag (paper, fakes) are trusted.
    """
    return isinstance(wallet, dict) and wallet.get("positions_known", True) is not False


BACKOFF_HTTP_STATUSES = frozenset({418, 429})


def is_backoff_status(status: int) -> bool:
    """Transport error, 5xx, or Binance rate limit / IP ban: stop and back off."""
    return status == 0 or status >= 500 or status in BACKOFF_HTTP_STATUSES


def retry_after_sec(body: Any) -> float | None:
    """Retry-After (seconds) captured from a 418/429 response, if any."""
    if not isinstance(body, dict):
        return None
    try:
        value = float(body.get("retry_after_sec"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


class ExchangeHTTPError(RuntimeError):
    """Non-2xx answer from a read endpoint (keeps the status for backoff)."""

    def __init__(self, what: str, status: int, body: Any) -> None:
        super().__init__(f"{what} HTTP {status}: {body}")
        self.status = int(status)
        self.body = body

    @property
    def backoff(self) -> bool:
        return is_backoff_status(self.status)

    @property
    def retry_after(self) -> float | None:
        return retry_after_sec(self.body)


class PositionsUnknown(RuntimeError):
    """The exchange answered without a positions list; do not treat as flat."""


def send_status_unknown(status: int, data: Any) -> bool:
    """Transport error, 5xx, or Binance 'send status unknown' codes."""
    if status == 0 or status >= 500:
        return True
    return isinstance(data, dict) and data.get("code") in SEND_UNKNOWN_CODES


def order_not_found(data: Any) -> bool:
    return isinstance(data, dict) and data.get("code") == -2013


def order_is_filled(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    status = str(body.get("status") or "")
    return order_executed_qty(body) > 0 or status in FILLED_ORDER_STATUSES


def is_real_fill(execution: ExecutionResult | None) -> bool:
    """True only for paper fills or Binance orders with executedQty > 0."""
    if execution is None:
        return False
    if execution.status == "paper_recorded":
        return True
    if execution.status not in {"filled", "accepted"}:
        return False
    body = _body(execution)
    if execution.status == "filled" and not isinstance(body, dict):
        return True
    if order_is_filled(body):
        return True
    if execution.status != "filled":
        return False
    try:
        return float(_detail(execution).get("filled_qty") or 0.0) > 0
    except (TypeError, ValueError):
        return False


def fill_qty(execution: ExecutionResult | None, fallback: float = 0.0) -> float:
    if execution is None:
        return 0.0
    if execution.status == "paper_recorded":
        return fallback
    qty = order_executed_qty(_body(execution))
    if qty > 0:
        return qty
    try:
        extra = float(_detail(execution).get("filled_qty") or 0.0)
    except (TypeError, ValueError):
        extra = 0.0
    if extra > 0:
        return extra
    return fallback if execution.status == "filled" else 0.0


def fill_price(execution: ExecutionResult | None, fallback: float | None) -> float | None:
    if execution is None:
        return fallback
    px = order_fill_price(_body(execution))
    if px is not None:
        return px
    try:
        extra = float(_detail(execution).get("fill_price") or 0.0)
    except (TypeError, ValueError):
        extra = 0.0
    if extra > 0:
        return extra
    return fallback


class PaperBroker:
    venue = "paper"

    def submit(self, intent: TradeIntent) -> ExecutionResult:
        blocked = blocked_execution(intent, self.venue)
        if blocked is not None:
            return blocked
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
        self._fill_poll_attempts = 4
        self._fill_poll_sleep = 0.25
        # Plain-BUY covers whose submit timed out: symbol -> (cid, monotonic).
        self._unknown_covers: dict[str, tuple[str, float]] = {}
        self.cover_unknown_grace_sec = 300.0
        # Optional ledger lookup (survives restarts): symbol -> cover cids
        # still in submit_unknown.
        # Items are cids or (cid, age_sec) pairs.
        self.unknown_cover_cids_fn: Callable[[str], list[Any]] | None = None
        # A ledger cover row this old can no longer be in flight: cancel by
        # cid and settle it via settle_unknown_cover_fn(cid, status).
        self.cover_unknown_max_age_sec = 72 * 3600.0
        self.settle_unknown_cover_fn: Callable[[str, str], Any] | None = None

    def sync_time(self) -> int:
        status, data = self._request("GET", "/fapi/v1/time", signed=False)
        if status == 200 and isinstance(data, dict) and data.get("serverTime"):
            server = int(data["serverTime"])
            self._time_offset_ms = server - int(time.time() * 1000)
            self._time_synced = True
            return server
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
            if status in BACKOFF_HTTP_STATUSES:
                raw_retry = exc.headers.get("Retry-After") if exc.headers is not None else None
                if not isinstance(data, dict):
                    data = {"body": data}
                try:
                    data["retry_after_sec"] = float(raw_retry) if raw_retry is not None else None
                except (TypeError, ValueError):
                    data["retry_after_sec"] = None
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
        from_account = True
        if not (200 <= status < 300) or not isinstance(data, dict) or data.get("code"):
            account_status, account_body = status, data
            from_account = False
            status, data = self._request("GET", "/fapi/v2/balance", signed=True)
        if not (200 <= status < 300):
            raise RuntimeError(f"binance wallet HTTP {status}: {data}")
        wallet = parse_usdt_wallet(data)
        # /fapi/v2/balance has no positions: an empty list there is "unknown".
        wallet["positions_known"] = bool(
            from_account and isinstance(data, dict) and isinstance(data.get("positions"), list)
        )
        if not from_account:
            log_event(
                "wallet_positions_unknown",
                account_http_status=account_status,
                account_body=account_body,
            )
        wallet["http_status"] = status
        wallet["venue"] = self.venue
        wallet["base"] = self.base_url
        self._wallet_cache = wallet
        self._wallet_ts = now
        return wallet

    def invalidate_wallet(self) -> None:
        self._wallet_cache = None
        self._wallet_ts = 0.0

    def filters_for(self, symbol: str) -> dict[str, str]:
        if not self._filters_loaded:
            status, data = self._request("GET", "/fapi/v1/exchangeInfo")
            if 200 <= status < 300 and isinstance(data, dict) and data.get("symbols"):
                self._filters = parse_symbol_filters(data)
                self._filters_loaded = True
        return self._filters.get(symbol.upper(), {})

    def _live_position(self, symbol: str) -> dict[str, Any] | None:
        """Fresh exchange position for symbol, or None if flat / missing."""
        self.invalidate_wallet()
        wallet = self.fetch_wallet(ttl=0)
        if not wallet_positions_known(wallet):
            raise PositionsUnknown(f"wallet without positions for {symbol}")
        sym = symbol.upper()
        for row in wallet.get("positions") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol") or "").upper() != sym:
                continue
            size = float(row.get("size") or 0.0)
            side = str(row.get("side") or "")
            if size > 0 and side in {"LONG", "SHORT"}:
                return row
        return None

    def _align_reduce_close(self, intent: TradeIntent) -> TradeIntent | ExecutionResult:
        """Force close side/qty from live exchange position (fixes stale SHORT/LONG)."""
        try:
            pos = self._live_position(intent.symbol)
        except PositionsUnknown:
            return _positions_unknown_result(intent, self.venue)
        if pos is None:
            return ExecutionResult(
                status="rejected",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=intent.reduce_only,
                detail={
                    "error": "no_position_to_reduce",
                    "http_status": 400,
                    "body": {"code": -2022, "msg": "ReduceOnly Order is rejected."},
                },
            )
        side = str(pos["side"])
        size = float(pos["size"])
        order_side = close_side_for_position(side)
        return replace(intent, order_side=order_side, qty=size)

    def _cover_short_without_reduce_only(
        self, intent: TradeIntent, qty: float
    ) -> ExecutionResult:
        """Binance testnet often rejects BUY+reduceOnly (-2022) even on a real SHORT.

        Cover with a plain BUY sized to the live short. Qty is clamped so we do not flip long.
        """
        try:
            pos = self._live_position(intent.symbol)
        except PositionsUnknown:
            return _positions_unknown_result(intent, self.venue)
        if pos is None or str(pos.get("side")) != "SHORT":
            return ExecutionResult(
                status="rejected",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=False,
                detail={
                    "error": "short_already_flat",
                    "http_status": 400,
                    "body": {"code": -2022, "msg": "ReduceOnly Order is rejected."},
                    "fallback": "cover_without_reduce_only",
                },
            )
        live_qty = float(pos["size"])
        qty = min(qty, live_qty) if qty > 0 else live_qty
        flt = self.filters_for(intent.symbol)
        if flt.get("stepSize"):
            qty = round_to_step(qty, flt["stepSize"])
        min_qty = float(flt.get("minQty") or 0.0)
        if qty <= 0 or (min_qty and qty < min_qty):
            return ExecutionResult(
                status="rejected",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=False,
                detail={"error": "qty below LOT_SIZE", "qty": qty, "filters": flt},
            )
        blocked_by = self._cover_in_flight(intent.symbol)
        if blocked_by is not None:
            log_event("cover_blocked_submit_unknown", symbol=intent.symbol, pending_cid=blocked_by)
            return ExecutionResult(
                status="rejected",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=False,
                detail={
                    "error": "cover_submit_unknown",
                    "pending_client_order_id": blocked_by,
                    "fallback": "cover_without_reduce_only",
                },
            )
        cid = client_order_id(intent.symbol, "flatten")
        params: dict[str, Any] = {
            "symbol": intent.symbol,
            "side": "BUY",
            "type": "MARKET",
            "quantity": format_binance_decimal(qty),
            "newClientOrderId": cid,
        }
        self.cancel_open_orders(intent.symbol)
        status, data = self._request("POST", "/fapi/v1/order", params=params, signed=True)
        if send_status_unknown(status, data):
            # A timeout / 5xx does not mean the exchange refused the order.
            # Ask by clientOrderId before calling it rejected.
            # The cover was sent under its own `cid`, not the intent's id.
            outcome, queried = self._resolve_submit(intent.symbol, cid)
            if outcome == "found":
                data = {**queried, "resolved_after": {"http_status": status, "body": data}}
                status = 200
            elif outcome == "absent":
                # A few quick -2013 right after a timeout / 5xx do not prove the
                # order was never accepted (Binance can lag). Keep it
                # submit_unknown: it counts in the hourly cap and the
                # reconciler settles it after its grace window.
                self.invalidate_wallet()
                self._unknown_covers[intent.symbol.upper()] = (cid, time.monotonic())
                return ExecutionResult(
                    status="submit_unknown",
                    venue=self.venue,
                    client_order_id=cid,
                    reduce_only=False,
                    detail={
                        "http_status": status,
                        "body": data,
                        "error": "submit_unknown",
                        "resolved": "not_found_yet",
                        "fallback": "cover_without_reduce_only",
                        "original_client_order_id": intent.client_order_id,
                    },
                )
            else:
                self.invalidate_wallet()
                self._unknown_covers[intent.symbol.upper()] = (cid, time.monotonic())
                return ExecutionResult(
                    status="submit_unknown",
                    venue=self.venue,
                    client_order_id=cid,
                    reduce_only=False,
                    detail={
                        "http_status": status,
                        "body": data,
                        "error": "submit_unknown",
                        "fallback": "cover_without_reduce_only",
                        "original_client_order_id": intent.client_order_id,
                    },
                )
        ok = 200 <= status < 300
        if not ok:
            return ExecutionResult(
                status="rejected",
                venue=self.venue,
                client_order_id=cid,
                reduce_only=False,
                detail={
                    "http_status": status,
                    "body": data,
                    "fallback": "cover_without_reduce_only",
                    "original_client_order_id": intent.client_order_id,
                },
            )
        filled_qty = order_executed_qty(data)
        fill_px = order_fill_price(data)
        if not order_is_filled(data):
            data = self._poll_order(intent.symbol, cid, data)
            filled_qty = order_executed_qty(data)
            fill_px = order_fill_price(data)
        self.invalidate_wallet()
        if order_is_filled(data):
            return ExecutionResult(
                status="filled",
                venue=self.venue,
                client_order_id=cid,
                reduce_only=False,
                detail={
                    "http_status": status,
                    "body": data,
                    "filled_qty": filled_qty,
                    "fill_price": fill_px,
                    "fallback": "cover_without_reduce_only",
                    "original_client_order_id": intent.client_order_id,
                },
            )
        self.cancel_order(intent.symbol, cid)
        self.invalidate_wallet()
        return ExecutionResult(
            status="unfilled",
            venue=self.venue,
            client_order_id=cid,
            reduce_only=False,
            detail={
                "http_status": status,
                "body": data,
                "error": "not_filled",
                "fallback": "cover_without_reduce_only",
            },
        )

    def _cover_in_flight(self, symbol: str) -> str | None:
        """cid of an earlier plain-BUY cover that may still fill, else None.

        A second cover sent while the first is in submit_unknown can flip the
        book long if both land.
        """
        sym = symbol.upper()
        cids: dict[str, float | None] = {}
        ages: dict[str, float] = {}
        mem = self._unknown_covers.get(sym)
        if mem is not None:
            cids[mem[0]] = mem[1]
        if self.unknown_cover_cids_fn is not None:
            try:
                for item in self.unknown_cover_cids_fn(sym) or []:
                    if isinstance(item, (tuple, list)):
                        cid = str(item[0])
                        age = item[1] if len(item) > 1 else None
                        if age is not None:
                            ages[cid] = float(age)
                    else:
                        cid = str(item)
                    cids.setdefault(cid, None)
            except Exception as exc:  # noqa: BLE001
                log_event("cover_guard_error", symbol=sym, error=f"{type(exc).__name__}: {exc}")
        for cid, since in cids.items():
            if ages.get(cid, 0.0) >= self.cover_unknown_max_age_sec:
                self._expire_unknown_cover(sym, cid, ages[cid])
                continue
            status, body = self.query_order(sym, cid)
            if 200 <= status < 300 and isinstance(body, dict) and body.get("orderId") is not None:
                if str(body.get("status") or "").upper() in {"NEW", "PARTIALLY_FILLED"}:
                    return cid
                # Final on the exchange; the live position read that follows
                # already reflects it.
                if mem is not None and mem[0] == cid:
                    self._unknown_covers.pop(sym, None)
                continue
            if order_not_found(body) and since is not None and time.monotonic() - since >= self.cover_unknown_grace_sec:
                # Never surfaced within the grace window: cancel by cid in case
                # it still does, then let a new cover through.
                self.cancel_order(sym, cid)
                self._unknown_covers.pop(sym, None)
                continue
            # Not found yet (exchange lag), or the query itself failed. Ledger
            # rows without a local timestamp wait for the reconciler to settle.
            return cid
        return None

    def _expire_unknown_cover(self, symbol: str, cid: str, age_sec: float) -> None:
        """Terminal path for a cover that stayed submit_unknown past the max age.

        A plain MARKET BUY does not rest for days: cancel by cid in case it
        somehow does, settle the ledger row as rejected and stop blocking.
        """
        cancel = self.cancel_order(symbol, cid)
        settled: Any = None
        if self.settle_unknown_cover_fn is not None:
            try:
                settled = self.settle_unknown_cover_fn(cid, "rejected")
            except Exception as exc:  # noqa: BLE001
                settled = f"{type(exc).__name__}: {exc}"
        mem = self._unknown_covers.get(symbol)
        if mem is not None and mem[0] == cid:
            self._unknown_covers.pop(symbol, None)
        log_event(
            "cover_unknown_expired",
            symbol=symbol,
            cid=cid,
            age_sec=round(age_sec),
            cancel_status=cancel.get("http_status") if isinstance(cancel, dict) else None,
            settled=settled,
        )

    def submit(self, intent: TradeIntent) -> ExecutionResult:
        with order_lock:
            return self._submit_locked(intent)

    def _submit_locked(self, intent: TradeIntent) -> ExecutionResult:
        blocked = blocked_execution(intent, self.venue)
        if blocked is not None:
            return blocked
        blind_close = False
        limits_cleared = False
        if intent.reduce_only and intent.action == "close" and intent.entry_type == "MARKET":
            # Drop resting entry limits before reading the live size: an entry
            # that fills between that read and the close would outgrow the
            # close qty, and the "full" fill would then strip the remainder's
            # stop. Stops are left alone (a blind close keeps them too).
            self.cancel_open_limits(intent.symbol)
            limits_cleared = True
            aligned = self._align_reduce_close(intent)
            if isinstance(aligned, ExecutionResult):
                if not (_is_positions_unknown(aligned) and blind_close_allowed(intent)):
                    return aligned
                # Positions unknown (balance-only wallet) but a protective exit
                # is due: a reduce-only SELL cannot open or flip a position, so
                # send it with the last-known qty (worst case -2022 when flat).
                blind_close = True
                log_event(
                    "blind_reduce_only_close",
                    symbol=intent.symbol,
                    qty=intent.qty,
                    risk_event=intent.risk_event,
                )
            else:
                intent = aligned
        qty = intent.qty
        price = intent.limit_price
        flt = self.filters_for(intent.symbol)
        if flt:
            if flt.get("stepSize"):
                qty = round_to_step(qty, flt["stepSize"])
            if price is not None and flt.get("tickSize"):
                price = round_to_step(price, flt["tickSize"])
                if intent.entry_type != "MARKET":
                    price = gtx_inside_price(price, intent.order_side, flt["tickSize"])
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
        if blind_close:
            # Keep the exchange stop until the close is known to have flattened.
            pass
        elif intent.reduce_only:
            # A reduce-only close and the closePosition stop can coexist: drop
            # only resting limits now; the stop goes after a confirmed full
            # fill, so a rejected / unknown / partial close stays protected.
            if not limits_cleared:
                self.cancel_open_limits(intent.symbol)
        elif intent.entry_type == "MARKET":
            self.cancel_open_orders(intent.symbol)
        else:
            self.cancel_open_limits(intent.symbol)
        status, data = self._request("POST", "/fapi/v1/order", params=params, signed=True)
        if send_status_unknown(status, data):
            # A timeout / 5xx does not mean the exchange refused the order.
            # Ask by clientOrderId before calling it rejected.
            outcome, queried = self._resolve_submit(intent.symbol, intent.client_order_id)
            if outcome == "found":
                data = {**queried, "resolved_after": {"http_status": status, "body": data}}
                status = 200
            elif outcome == "absent":
                # A few quick -2013 right after a timeout / 5xx do not prove the
                # order was never accepted (Binance can lag). Keep it
                # submit_unknown: it counts in the hourly cap and the
                # reconciler settles it after its grace window.
                self.invalidate_wallet()
                return ExecutionResult(
                    status="submit_unknown",
                    venue=self.venue,
                    client_order_id=intent.client_order_id,
                    reduce_only=intent.reduce_only,
                    detail={"http_status": status, "body": data, "error": "submit_unknown", "resolved": "not_found_yet"},
                )
            else:
                self.invalidate_wallet()
                return ExecutionResult(
                    status="submit_unknown",
                    venue=self.venue,
                    client_order_id=intent.client_order_id,
                    reduce_only=intent.reduce_only,
                    detail={"http_status": status, "body": data, "error": "submit_unknown"},
                )
        ok = 200 <= status < 300
        if not ok:
            # Testnet (and some one-way states) reject BUY+reduceOnly on a real SHORT
            # with -2022. Cover via plain BUY clamped to abs(live short).
            if (
                intent.reduce_only
                and intent.action == "close"
                and intent.order_side == "BUY"
                and intent.entry_type == "MARKET"
                and is_reduce_only_rejected(data)
            ):
                return self._cover_short_without_reduce_only(intent, qty)
            return ExecutionResult(
                status="rejected",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=intent.reduce_only,
                detail={"http_status": status, "body": data},
            )
        filled_qty = order_executed_qty(data)
        fill_px = order_fill_price(data)
        if not order_is_filled(data) and intent.entry_type == "MARKET":
            data = self._poll_order(intent.symbol, intent.client_order_id, data)
            filled_qty = order_executed_qty(data)
            fill_px = order_fill_price(data)
        self.invalidate_wallet()
        if order_is_filled(data):
            detail = {
                "http_status": status,
                "body": data,
                "filled_qty": filled_qty,
                "fill_price": fill_px,
            }
            if blind_close:
                detail["blind_close"] = True
                # Reduce-only fills at most the position: less than asked means
                # the book is now flat, so the closePosition stop can go. Else a
                # remainder may exist and keeps its stop.
                if 0 < filled_qty < qty:
                    self.cancel_open_orders(intent.symbol)
            elif intent.reduce_only and _close_fully_filled(data, filled_qty, qty, flt):
                self.cancel_open_orders(intent.symbol)
                detail["stop_cleared"] = True
            elif intent.reduce_only:
                # Partial close: the remainder keeps its closePosition stop.
                detail["stop_kept"] = "partial_close"
                log_event("close_partial_stop_kept", symbol=intent.symbol, filled_qty=filled_qty, qty=qty)
            return ExecutionResult(
                status="filled",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=intent.reduce_only,
                detail=detail,
            )
        if intent.entry_type == "MARKET":
            self.cancel_order(intent.symbol, intent.client_order_id)
            self.invalidate_wallet()
            return ExecutionResult(
                status="unfilled",
                venue=self.venue,
                client_order_id=intent.client_order_id,
                reduce_only=intent.reduce_only,
                detail={"http_status": status, "body": data, "error": "not_filled"},
            )
        return ExecutionResult(
            status="working",
            venue=self.venue,
            client_order_id=intent.client_order_id,
            reduce_only=intent.reduce_only,
            detail={"http_status": status, "body": data},
        )

    def _poll_order(self, symbol: str, client_order_id: str, last: Any) -> Any:
        data = last
        for _ in range(max(0, int(self._fill_poll_attempts))):
            if order_is_filled(data):
                return data
            if self._fill_poll_sleep:
                time.sleep(self._fill_poll_sleep)
            status, queried = self._request(
                "GET",
                "/fapi/v1/order",
                params={"symbol": symbol, "origClientOrderId": client_order_id},
                signed=True,
            )
            if 200 <= status < 300:
                data = queried
        return data

    def _resolve_submit(self, symbol: str, client_order_id: str) -> tuple[str, Any]:
        """found (order body) / absent (-2013 on every try) / unknown."""
        last = "unknown"
        body: Any = None
        for attempt in range(max(1, int(self._fill_poll_attempts))):
            if attempt and self._fill_poll_sleep:
                time.sleep(self._fill_poll_sleep)
            status, body = self.query_order(symbol, client_order_id)
            if 200 <= status < 300 and isinstance(body, dict) and body.get("orderId") is not None:
                return "found", body
            last = "absent" if order_not_found(body) else "unknown"
        return last, body

    def query_order(
        self,
        symbol: str,
        client_order_id: str | None = None,
        *,
        order_id: Any = None,
        timeout: float = 10.0,
    ) -> tuple[int, Any]:
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        else:
            params["origClientOrderId"] = client_order_id
        return self._request("GET", "/fapi/v1/order", params=params, signed=True, timeout=timeout)

    def user_trades(
        self,
        symbol: str,
        *,
        order_id: Any = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        """Account trades (price, qty, commission, realizedPnl). Raises on HTTP error."""
        params: dict[str, Any] = {"symbol": symbol.upper(), "limit": int(limit)}
        if order_id is not None:
            params["orderId"] = order_id
        else:
            if start_ms is not None:
                params["startTime"] = int(start_ms)
            if end_ms is not None:
                params["endTime"] = int(end_ms)
        status, data = self._request(
            "GET", "/fapi/v1/userTrades", params=params, signed=True, timeout=timeout
        )
        if not (200 <= status < 300) or not isinstance(data, list):
            raise ExchangeHTTPError("userTrades", status, data)
        return [row for row in data if isinstance(row, dict)]

    def funding_income(
        self, symbol: str, start_ms: int, end_ms: int, *, timeout: float = 10.0
    ) -> float | None:
        """Sum of FUNDING_FEE income for symbol in [start, end]; None if unavailable.

        Raises ExchangeHTTPError on a transport error / 5xx / 418 / 429 so the
        caller can back off instead of hammering a throttled API.
        """
        status, data = self._request(
            "GET",
            "/fapi/v1/income",
            params={
                "symbol": symbol.upper(),
                "incomeType": "FUNDING_FEE",
                "startTime": int(start_ms),
                "endTime": int(end_ms),
                "limit": 1000,
            },
            signed=True,
            timeout=timeout,
        )
        if is_backoff_status(status):
            raise ExchangeHTTPError("income", status, data)
        if not (200 <= status < 300) or not isinstance(data, list):
            return None
        total = 0.0
        for row in data:
            if not isinstance(row, dict):
                continue
            try:
                total += float(row.get("income") or 0.0)
            except (TypeError, ValueError):
                continue
        return total

    def cancel_order(
        self, symbol: str, client_order_id: str, *, timeout: float = 10.0
    ) -> dict[str, Any]:
        status, data = self._request(
            "DELETE",
            "/fapi/v1/order",
            params={"symbol": symbol, "origClientOrderId": client_order_id},
            signed=True,
            timeout=timeout,
        )
        return {"http_status": status, "body": data}

    def cancel_open_orders(self, symbol: str) -> dict[str, Any]:
        status, data = self._request(
            "DELETE",
            "/fapi/v1/allOpenOrders",
            params={"symbol": symbol.upper()},
            signed=True,
        )
        return {"http_status": status, "body": data}

    def cancel_open_limits(self, symbol: str) -> list[dict[str, Any]]:
        status, data = self._request(
            "GET",
            "/fapi/v1/openOrders",
            params={"symbol": symbol.upper()},
            signed=True,
        )
        if not (200 <= status < 300) or not isinstance(data, list):
            return []
        cancelled: list[dict[str, Any]] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            typ = str(row.get("type") or "")
            tif = str(row.get("timeInForce") or "")
            if typ not in LIMIT_ORDER_TYPES and tif != "GTX":
                continue
            oid = row.get("orderId")
            params: dict[str, Any] = {"symbol": symbol.upper()}
            if oid is not None:
                params["orderId"] = oid
            else:
                params["origClientOrderId"] = row.get("clientOrderId")
            c_status, c_body = self._request("DELETE", "/fapi/v1/order", params=params, signed=True)
            cancelled.append({"http_status": c_status, "body": c_body})
        return cancelled

    def cancel_all_open_orders(
        self, symbols: list[str] | None = None, *, exclude: set[str] | None = None
    ) -> list[dict[str, Any]]:
        names = [s.upper() for s in (symbols or [])]
        if not names:
            status, data = self._request("GET", "/fapi/v1/openOrders", signed=True)
            if 200 <= status < 300 and isinstance(data, list):
                names = sorted(
                    {
                        str(row.get("symbol") or "").upper()
                        for row in data
                        if isinstance(row, dict) and row.get("symbol")
                    }
                )
        skip = {s.upper() for s in (exclude or set())}
        return [self.cancel_open_orders(symbol) for symbol in names if symbol not in skip]

    def cancel_working_entries(self) -> list[dict[str, Any]]:
        """Cancel resting LIMIT/GTX entries. Leave STOP_MARKET protection in place."""
        status, data = self._request("GET", "/fapi/v1/openOrders", signed=True)
        if not (200 <= status < 300) or not isinstance(data, list):
            return []
        symbols = sorted(
            {
                str(row.get("symbol") or "").upper()
                for row in data
                if isinstance(row, dict) and row.get("symbol")
            }
        )
        cancelled: list[dict[str, Any]] = []
        for symbol in symbols:
            cancelled.extend(self.cancel_open_limits(symbol))
        return cancelled

    def _close_stop_open(self, symbol: str) -> bool:
        return bool(self._close_stops(symbol))

    def _open_orders(self, symbol: str) -> list[dict[str, Any]] | None:
        status, data = self._request(
            "GET",
            "/fapi/v1/openOrders",
            params={"symbol": symbol.upper()},
            signed=True,
        )
        if not (200 <= status < 300) or not isinstance(data, list):
            return None
        return [row for row in data if isinstance(row, dict)]

    @staticmethod
    def _close_stops_from(rows: list[dict[str, Any]], order_side: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("type") or "") not in CLOSE_STOP_TYPES:
                continue
            flag = row.get("closePosition")
            if not (flag is True or str(flag).lower() == "true"):
                continue
            side = str(row.get("side") or "").upper()
            if order_side and side and side != order_side.upper():
                continue
            out.append(row)
        return out

    @staticmethod
    def _temp_stops_from(rows: list[dict[str, Any]], order_side: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("type") or "") not in CLOSE_STOP_TYPES:
                continue
            if not str(row.get("clientOrderId") or "").startswith(TEMP_STOP_PREFIX):
                continue
            side = str(row.get("side") or "").upper()
            if order_side and side and side != order_side.upper():
                continue
            out.append(row)
        return out

    def _close_stops(self, symbol: str, order_side: str | None = None) -> list[dict[str, Any]] | None:
        """Open closePosition STOP/STOP_MARKET orders (None if the query failed)."""
        rows = self._open_orders(symbol)
        if rows is None:
            return None
        return self._close_stops_from(rows, order_side)

    def _stop_matches(self, row: dict[str, Any], target: float, tick: float) -> bool:
        try:
            have = float(row.get("stopPrice"))
        except (TypeError, ValueError):
            return True  # cannot compare: do not churn the stop
        if have <= 0:
            return True
        tol = tick / 2.0 if tick > 0 else max(1e-9, abs(target) * 1e-9)
        return abs(have - target) <= tol

    def _cancel_row(self, symbol: str, row: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if row.get("orderId") is not None:
            params["orderId"] = row.get("orderId")
        else:
            params["origClientOrderId"] = row.get("clientOrderId")
        status, data = self._request("DELETE", "/fapi/v1/order", params=params, signed=True)
        return {"http_status": status, "body": data}

    def _place_temp_stop(self, symbol: str, stop_price: float, order_side: str) -> dict[str, Any] | None:
        """Reduce-only STOP_MARKET with the live position qty (None: no position read).

        Unlike closePosition stops it is not limited to one per direction
        (-4130), so it can sit next to the old closePosition stop while that
        one is swapped.
        """
        try:
            pos = self._live_position(symbol)
        except PositionsUnknown:
            return None
        if pos is None:
            return None
        want = "LONG" if order_side.upper() == "SELL" else "SHORT"
        if str(pos.get("side") or "").upper() != want:
            return None
        qty = abs(float(pos.get("size") or 0.0))
        flt = self.filters_for(symbol)
        if flt.get("stepSize"):
            qty = round_to_step(qty, flt["stepSize"])
        if qty <= 0:
            return None
        price = stop_price
        if flt.get("tickSize"):
            price = round_to_step(stop_price, flt["tickSize"])
        cid = client_order_id(symbol, "stop_tmp")
        params = {
            "symbol": symbol.upper(),
            "side": order_side,
            "type": "STOP_MARKET",
            "stopPrice": format_binance_decimal(price),
            "quantity": format_binance_decimal(qty),
            "reduceOnly": "true",
            "workingType": "MARK_PRICE",
            "newClientOrderId": cid,
        }
        status, data = self._request("POST", "/fapi/v1/order", params=params, signed=True)
        self.invalidate_wallet()
        return {"http_status": status, "body": data, "client_order_id": cid, "qty": qty}

    def _drop_temps(self, symbol: str, temps: list[dict[str, Any]], keep: str | None = None) -> list[dict[str, Any]]:
        return [
            self._cancel_row(symbol, row)
            for row in temps
            if keep is None or str(row.get("clientOrderId") or "") != keep
        ]

    def ensure_stop_market(
        self,
        symbol: str,
        *,
        stop_price: float,
        order_side: str = "SELL",
        tighten_only: bool = False,
    ) -> dict[str, Any]:
        """Make the exchange's close-all stop sit at `stop_price`.

        A closePosition stop at another price (left from an earlier position
        or an older plan) is replaced. Binance keeps one closePosition stop
        per direction (-4130), so the swap is covered by a temporary
        reduce-only STOP_MARKET for the live qty at the new price: temp first,
        then cancel old, place new, cancel temp. If the new closePosition stop
        is refused, the temp stays as the protection. Without a temp (no
        position read, or the temp refused) the old flow runs: cancel, place,
        restore the old stop on refusal. When both the new stop and the
        restore are refused, `unprotected` is set and the caller must close.

        tighten_only: an existing stop that is already tighter (higher for a
        SELL stop) is kept, so a recomputed ATR stop never loosens it.
        """
        with order_lock:
            rows = self._open_orders(symbol)
            if rows is None:
                # Could not list orders: try to place; a duplicate is refused.
                return self.place_stop_market(symbol, stop_price=stop_price, order_side=order_side)
            stops = self._close_stops_from(rows, order_side)
            temps = self._temp_stops_from(rows, order_side)
            if not stops:
                res = self.place_stop_market(symbol, stop_price=stop_price, order_side=order_side)
                if temps and _http_ok(res):
                    res["temp_cancels"] = self._drop_temps(symbol, temps)
                return res
            flt = self.filters_for(symbol)
            tick = float(flt.get("tickSize") or 0.0) if flt else 0.0
            if tick <= 0:
                # No tick size (exchangeInfo missing): the unrounded target never
                # equals the exchange's price, and sending it is refused for
                # precision. Replacing would cancel + fail + restore every cycle,
                # each time leaving the long bare. Keep the stop that is there.
                log_event("stop_kept_unverified", symbol=symbol, want=stop_price, have=[r.get("stopPrice") for r in stops])
                return {"http_status": 200, "body": {"skipped": "stop_exists", "unverified": "no_tick_size"}, "stop_price": stop_price}
            target = round_to_step(stop_price, flt["tickSize"])
            if any(self._stop_matches(row, target, tick) for row in stops):
                res = {"http_status": 200, "body": {"skipped": "stop_exists"}, "stop_price": stop_price}
                if temps:
                    res["temp_cancels"] = self._drop_temps(symbol, temps)
                return res
            if tighten_only:
                tighter = _tighter_stop_prices(stops, target, tick, order_side)
                if tighter:
                    log_event("stop_kept_tighter", symbol=symbol, want=target, have=tighter)
                    return {
                        "http_status": 200,
                        "body": {"skipped": "stop_tighter"},
                        "stop_price": stop_price,
                        "kept_stop_price": tighter[0],
                    }
            old_prices = [row.get("stopPrice") for row in stops]
            temp_cid: str | None = None
            temp: dict[str, Any] | None = None
            reuse = [
                r for r in temps
                if self._stop_matches(r, target, tick) and r.get("stopPrice") is not None
            ]
            if reuse:
                temp_cid = str(reuse[0].get("clientOrderId") or "")
            else:
                temp = self._place_temp_stop(symbol, target, order_side)
                if temp is not None and _http_ok(temp):
                    temp_cid = str(temp["client_order_id"])
            cancels = [self._cancel_row(symbol, row) for row in stops]
            res = self.place_stop_market(symbol, stop_price=stop_price, order_side=order_side)
            res["replaced"] = {"old_stop_prices": old_prices, "cancels": cancels}
            if temp is not None:
                res["temp_stop"] = temp
            if _http_ok(res):
                if temp_cid:
                    res["temp_cancel"] = self.cancel_order(symbol, temp_cid)
                others = [r for r in temps if str(r.get("clientOrderId") or "") != temp_cid]
                if others:
                    res["temp_cancels"] = self._drop_temps(symbol, others)
                log_event("stop_replaced", symbol=symbol, old=old_prices, new=target, covered=bool(temp_cid))
                return res
            if temp_cid:
                # The temporary reduce-only stop at the new price protects the
                # position; the next ensure turns it into a closePosition stop.
                res["protected_by"] = "temp_stop"
                res["temp_client_order_id"] = temp_cid
                log_event(
                    "stop_replace_temp_kept",
                    symbol=symbol,
                    old=old_prices,
                    new=target,
                    http_status=res.get("http_status"),
                    body=res.get("body"),
                )
                return res
            restore: dict[str, Any] | None = None
            try:
                old = float(old_prices[0])
            except (TypeError, ValueError, IndexError):
                old = 0.0
            if old > 0:
                restore = self.place_stop_market(symbol, stop_price=old, order_side=order_side)
            res["restored_old_stop"] = restore
            if restore is None or not _http_ok(restore):
                res["unprotected"] = True
            log_event(
                "stop_replace_failed",
                symbol=symbol,
                old=old_prices,
                new=target,
                http_status=res.get("http_status"),
                body=res.get("body"),
                restore=restore,
                unprotected=bool(res.get("unprotected")),
            )
            return res

    def place_stop_market(
        self,
        symbol: str,
        *,
        stop_price: float,
        order_side: str = "SELL",
    ) -> dict[str, Any]:
        flt = self.filters_for(symbol)
        price = stop_price
        if flt.get("tickSize"):
            price = round_to_step(stop_price, flt["tickSize"])
        if price <= 0:
            return {"http_status": 0, "body": {"error": "invalid stop_price"}}
        params = {
            "symbol": symbol.upper(),
            "side": order_side,
            "type": "STOP_MARKET",
            "stopPrice": format_binance_decimal(price),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
        }
        status, data = self._request("POST", "/fapi/v1/order", params=params, signed=True)
        self.invalidate_wallet()
        return {"http_status": status, "body": data}

    def flatten_all(self) -> dict[str, Any]:
        with order_lock:
            return self._flatten_all_locked()

    def _flatten_all_locked(self) -> dict[str, Any]:
        closes: list[dict[str, Any]] = []
        probe = self.fetch_wallet(ttl=0)
        if not wallet_positions_known(probe):
            # Without a positions list we cannot tell what to close. Do not
            # cancel first: that would strip the exchange stops and close
            # nothing. The caller falls back to the ledger's last-known longs.
            log_event("flatten_positions_unknown")
            return {"cancelled": [], "closes": closes, "wallet": probe, "error": "positions_unknown"}
        # Symbols with a position keep their closePosition stop: submit() drops
        # their limits before the close and the stop after a confirmed fill.
        positioned = {
            str(p.get("symbol") or "").upper()
            for p in probe.get("positions") or []
            if isinstance(p, dict) and float(p.get("size") or 0.0) > 0
        }
        cancelled = self.cancel_all_open_orders(exclude=positioned)
        wallet = self.fetch_wallet(ttl=0)
        if not wallet_positions_known(wallet):
            log_event("flatten_positions_unknown")
            return {"cancelled": cancelled, "closes": closes, "wallet": wallet, "error": "positions_unknown"}
        for pos in wallet.get("positions") or []:
            symbol = str(pos.get("symbol") or "")
            size = float(pos.get("size") or 0.0)
            side = str(pos.get("side") or "")
            if not symbol or size <= 0:
                continue
            intent = market_close_intent(
                symbol=symbol,
                qty=size,
                order_side="SELL" if side == "LONG" else "BUY",
                client_order_id=client_order_id(symbol, "flatten"),
                risk_event="flatten",
            )
            result = self.submit(intent)
            closes.append(
                {
                    "symbol": symbol,
                    "status": result.status,
                    "detail": result.detail,
                    "qty": size,
                    "side": side,
                    "client_order_id": intent.client_order_id,
                    "order_side": intent.order_side,
                }
            )
        after = self.fetch_wallet(ttl=0)
        return {"cancelled": cancelled, "closes": closes, "wallet": after}


# Exits that must not wait for a positions list: a reduce-only SELL is safe to
# send blind (it can only shrink a long; the exchange rejects it when flat).
BLIND_CLOSE_RISK_EVENTS = frozenset({"kill_switch", "daily_loss", "stop", "flatten"})


def _close_fully_filled(body: Any, filled_qty: float, qty: float, flt: dict[str, Any] | None) -> bool:
    """Reduce-only close done: FILLED, or executed within half a step of qty."""
    if isinstance(body, dict) and str(body.get("status") or "") == "FILLED" and filled_qty > 0:
        return True
    step = 0.0
    try:
        step = float((flt or {}).get("stepSize") or 0.0)
    except (TypeError, ValueError):
        step = 0.0
    return filled_qty > 0 and filled_qty >= qty - step / 2 - 1e-12


def _http_ok(res: Any) -> bool:
    status = res.get("http_status") if isinstance(res, dict) else None
    return isinstance(status, int) and 200 <= status < 300


def _tighter_stop_prices(
    stops: list[dict[str, Any]], target: float, tick: float, order_side: str
) -> list[float]:
    """Existing stop prices already tighter than target (SELL: higher, BUY: lower)."""
    tol = tick / 2.0 if tick > 0 else max(1e-9, abs(target) * 1e-9)
    out: list[float] = []
    for row in stops:
        try:
            have = float(row.get("stopPrice"))
        except (TypeError, ValueError):
            continue
        if have <= 0:
            continue
        if order_side.upper() == "SELL" and have > target + tol:
            out.append(have)
        elif order_side.upper() == "BUY" and have < target - tol:
            out.append(have)
    return sorted(out, reverse=order_side.upper() == "SELL")


def blind_close_allowed(intent: TradeIntent) -> bool:
    return (
        intent.reduce_only
        and intent.action == "close"
        and intent.entry_type == "MARKET"
        and intent.order_side == "SELL"
        and intent.qty > 0
        and intent.risk_event in BLIND_CLOSE_RISK_EVENTS
    )


def _is_positions_unknown(result: ExecutionResult) -> bool:
    detail = result.detail if isinstance(result.detail, dict) else {}
    return detail.get("error") == "positions_unknown"


def _positions_unknown_result(intent: TradeIntent, venue: str) -> ExecutionResult:
    return ExecutionResult(
        status="rejected",
        venue=venue,
        client_order_id=intent.client_order_id,
        reduce_only=intent.reduce_only,
        detail={"error": "positions_unknown"},
    )


class BinanceTestnetBroker(BinanceFuturesBroker):
    def __init__(self, api_key: str, api_secret: str, base_url: str) -> None:
        super().__init__(api_key, api_secret, base_url, live=False)
