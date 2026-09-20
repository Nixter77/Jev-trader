"""Unsigned Binance USD-M public market data.

Production REST (`fapi.binance.com`) and WS (`fstream.binance.com`) are allowed
**only** here, and only as security type NONE. Requests never send
`BINANCE_API_KEY`, `X-MBX-APIKEY`, or an HMAC `signature`. Signed trading stays
on testnet via `BinanceTestnetBroker`.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from jev_trader.features import parse_book
from jev_trader.http import ssl_context
from jev_trader.models import Candle, OrderBook

PUBLIC_REST_HOST = "fapi.binance.com"
PUBLIC_REST_BASE = "https://fapi.binance.com"
PUBLIC_WS_HOST = "fstream.binance.com"
PUBLIC_WS_BASE = "wss://fstream.binance.com"

TICKER_24HR_PATH = "/fapi/v1/ticker/24hr"
KLINES_PATH = "/fapi/v1/klines"
DEPTH_PATH = "/fapi/v1/depth"
SIGNED_ORDER_PATH = "/fapi/v1/order"

MINI_TICKER_STREAM = "!miniTicker@arr"
DEFAULT_KLINE_LIMIT = 240
DEFAULT_DEPTH_LIMIT = 5

_SIGNED_QUERY_KEYS = frozenset({"signature", "timestamp", "recvwindow"})
_APIKEY_HEADER = "x-mbx-apikey"


@dataclass(frozen=True)
class UniverseTicker:
    symbol: str
    last: float
    price_change_percent: float | None
    high_24h: float | None = None
    low_24h: float | None = None
    volume: float | None = None
    quote_volume: float | None = None
    close_time: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last": self.last,
            "price_change_percent": self.price_change_percent,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "close_time": self.close_time,
        }


@dataclass(frozen=True)
class ParsedKline:
    candle: Candle
    close_time: int
    closed: bool | None = None
    symbol: str | None = None


def public_ws_url(stream: str) -> str:
    stream = stream.lstrip("/")
    return f"{PUBLIC_WS_BASE}/ws/{stream}"


def public_kline_stream(symbol: str, interval: str = "5m") -> str:
    return f"{symbol.lower()}@kline_{interval}"


def _clean_query(query: dict[str, Any] | None) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in dict(query or {}).items():
        if key.lower() in _SIGNED_QUERY_KEYS:
            continue
        if value is None:
            continue
        cleaned[str(key)] = str(value)
    return cleaned


def build_public_get_request(
    path: str,
    query: dict[str, Any] | None = None,
) -> urllib.request.Request:
    """Unsigned GET against the public UM futures REST host.

    Environment API keys are ignored. Query keys used for HMAC trading
    (`signature`, `timestamp`, `recvWindow`) are stripped if present.
    """
    if not path.startswith("/"):
        path = "/" + path
    params = _clean_query(query)
    url = f"{PUBLIC_REST_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json"},
    )


def build_ticker_24hr_request(symbol: str | None = None) -> urllib.request.Request:
    query = {"symbol": symbol} if symbol else None
    return build_public_get_request(TICKER_24HR_PATH, query)


def build_klines_request(
    symbol: str,
    interval: str = "5m",
    limit: int = DEFAULT_KLINE_LIMIT,
) -> urllib.request.Request:
    return build_public_get_request(
        KLINES_PATH,
        {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)},
    )


def build_depth_request(
    symbol: str,
    limit: int = DEFAULT_DEPTH_LIMIT,
) -> urllib.request.Request:
    return build_public_get_request(
        DEPTH_PATH,
        {"symbol": symbol.upper(), "limit": int(limit)},
    )


def assert_unsigned_request(req: urllib.request.Request) -> None:
    parsed = urllib.parse.urlparse(req.full_url)
    qs = {k.lower(): v for k, v in urllib.parse.parse_qs(parsed.query).items()}
    if "signature" in qs:
        raise RuntimeError("public market request must not be signed")
    header_map = {str(key).lower(): str(value) for key, value in req.header_items()}
    if _APIKEY_HEADER in header_map:
        raise RuntimeError("public market request must not send X-MBX-APIKEY")
    leaked = (os.environ.get("BINANCE_API_KEY") or "").strip()
    secret = (os.environ.get("BINANCE_API_SECRET") or "").strip()
    header_values = list(header_map.values())
    query_values = [item for values in qs.values() for item in values]
    if leaked and (leaked in header_values or leaked in query_values):
        raise RuntimeError("public market request leaked BINANCE_API_KEY")
    if secret and (secret in header_values or secret in query_values):
        raise RuntimeError("public market request leaked BINANCE_API_SECRET")


def fetch_public_json(
    path: str,
    query: dict[str, Any] | None = None,
    *,
    timeout: float = 15.0,
    retries: int = 3,
) -> Any:
    req = build_public_get_request(path, query)
    assert_unsigned_request(req)
    last_exc: Exception | None = None
    attempts = max(1, int(retries))
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as resp:
                body = resp.read().decode("utf-8")
                status = getattr(resp, "status", 200)
                if status != 200:
                    raise RuntimeError(f"public GET {path} HTTP {status}")
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code < 500 or attempt + 1 >= attempts:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"public GET {path} HTTP {exc.code}: {body[:300]}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
        time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"public GET {path} failed: {last_exc}") from last_exc


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_ticker_item(item: dict[str, Any]) -> UniverseTicker:
    symbol = str(item.get("symbol") or item.get("s") or "")
    last_raw = item.get("lastPrice", item.get("c", item.get("close", item.get("markPrice"))))
    if last_raw is None or last_raw == "":
        raise ValueError(f"ticker missing last price: {item!r}")
    pct = item.get("priceChangePercent", item.get("P"))
    if pct is None or pct == "":
        open_px = _opt_float(item.get("openPrice", item.get("o")))
        last = float(last_raw)
        pct_val = ((last - open_px) / open_px * 100.0) if open_px else None
    else:
        pct_val = float(pct)
        last = float(last_raw)
    close_time = item.get("closeTime", item.get("E"))
    return UniverseTicker(
        symbol=symbol,
        last=last,
        price_change_percent=pct_val,
        high_24h=_opt_float(item.get("highPrice", item.get("h"))),
        low_24h=_opt_float(item.get("lowPrice", item.get("l"))),
        volume=_opt_float(item.get("volume", item.get("v"))),
        quote_volume=_opt_float(item.get("quoteVolume", item.get("q"))),
        close_time=None if close_time is None else int(close_time),
    )


def parse_ticker_24hr(payload: Any) -> list[UniverseTicker]:
    """Parse REST `/fapi/v1/ticker/24hr` or WS `!miniTicker@arr` payloads."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        data = payload.get("data", payload)
        if isinstance(data, list):
            items = data
        else:
            items = [data]
    elif isinstance(payload, list):
        items = payload
    else:
        raise TypeError(f"unexpected ticker payload type: {type(payload)!r}")
    rows: list[UniverseTicker] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = parse_ticker_item(item)
        if row.symbol:
            rows.append(row)
    return rows


def parse_rest_kline_row(row: Any) -> ParsedKline:
    if isinstance(row, dict):
        open_time = int(row.get("openTime", row.get("t")))
        close_time = int(row.get("closeTime", row.get("T", open_time)))
        candle = Candle(
            ts=open_time,
            open=float(row["open"] if "open" in row else row["o"]),
            high=float(row["high"] if "high" in row else row["h"]),
            low=float(row["low"] if "low" in row else row["l"]),
            close=float(row["close"] if "close" in row else row["c"]),
            volume=float(row["volume"] if "volume" in row else row["v"]),
        )
        closed = row.get("x")
        return ParsedKline(
            candle=candle,
            close_time=close_time,
            closed=None if closed is None else bool(closed),
            symbol=row.get("symbol") or row.get("s"),
        )
    open_time = int(row[0])
    return ParsedKline(
        candle=Candle(
            ts=open_time,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        ),
        close_time=int(row[6]),
        closed=None,
        symbol=None,
    )


def parse_rest_klines(payload: Any) -> list[ParsedKline]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("klines") or []
    return [parse_rest_kline_row(row) for row in payload]


def rest_closed_klines(
    rows: list[ParsedKline],
    *,
    now_ms: int | None = None,
) -> list[ParsedKline]:
    """Drop the still-forming REST bar. Last element is open until closeTime."""
    if not rows:
        return []
    cutoff = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return [row for row in rows if row.close_time < cutoff]


def parse_ws_kline_event(payload: dict[str, Any]) -> ParsedKline | None:
    """Parse a Binance `{symbol}@kline_5m` event. Combined-stream `{data}` ok."""
    data: Any = payload
    if "k" not in data and isinstance(payload.get("data"), dict):
        data = payload["data"]
    k = data.get("k") if isinstance(data, dict) else None
    if not isinstance(k, dict):
        return None
    candle = Candle(
        ts=int(k["t"]),
        open=float(k["o"]),
        high=float(k["h"]),
        low=float(k["l"]),
        close=float(k["c"]),
        volume=float(k["v"]),
    )
    return ParsedKline(
        candle=candle,
        close_time=int(k["T"]),
        closed=bool(k.get("x")),
        symbol=str(k.get("s") or data.get("s") or ""),
    )


def parse_depth(payload: dict[str, Any] | None) -> OrderBook | None:
    if not payload:
        return None
    return parse_book(payload)


def fetch_universe_summary(*, timeout: float = 15.0, retries: int = 3) -> list[UniverseTicker]:
    payload = fetch_public_json(TICKER_24HR_PATH, timeout=timeout, retries=retries)
    rows = parse_ticker_24hr(payload)
    if not rows:
        raise RuntimeError("empty universe summary from public ticker/24hr")
    return rows


def fetch_klines(
    symbol: str,
    *,
    interval: str = "5m",
    limit: int = DEFAULT_KLINE_LIMIT,
    timeout: float = 15.0,
    retries: int = 3,
) -> list[ParsedKline]:
    payload = fetch_public_json(
        KLINES_PATH,
        {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)},
        timeout=timeout,
        retries=retries,
    )
    rows = parse_rest_klines(payload)
    if not rows:
        raise RuntimeError(f"empty klines for {symbol}")
    return rows


def fetch_depth(
    symbol: str,
    *,
    limit: int = DEFAULT_DEPTH_LIMIT,
    timeout: float = 15.0,
    retries: int = 3,
) -> OrderBook | None:
    payload = fetch_public_json(
        DEPTH_PATH,
        {"symbol": symbol.upper(), "limit": int(limit)},
        timeout=timeout,
        retries=retries,
    )
    if not isinstance(payload, dict):
        return None
    return parse_depth(payload)


def universe_payload(rows: list[UniverseTicker]) -> dict[str, Any]:
    return {
        "ok": True,
        "type": "universe",
        "host": PUBLIC_REST_HOST,
        "path": TICKER_24HR_PATH,
        "authenticated": False,
        "count": len(rows),
        "tickers": [row.as_dict() for row in rows],
    }


def compact_universe_payload(
    rows: list[UniverseTicker],
    *,
    watch: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    top: int = 8,
) -> dict[str, Any]:
    """Terminal-sized universe view: count + watch + movers, not 700+ tickers."""
    by_symbol = {row.symbol: row for row in rows}
    watch_map: dict[str, Any] = {}
    for symbol in watch:
        row = by_symbol.get(symbol)
        watch_map[symbol] = (
            None
            if row is None
            else {"last": row.last, "price_change_percent": row.price_change_percent}
        )
    ranked = [row for row in rows if row.price_change_percent is not None]
    gainers = sorted(ranked, key=lambda row: float(row.price_change_percent or 0.0), reverse=True)[:top]
    losers = sorted(ranked, key=lambda row: float(row.price_change_percent or 0.0))[:top]

    def _move(row: UniverseTicker) -> dict[str, Any]:
        return {
            "symbol": row.symbol,
            "last": row.last,
            "price_change_percent": row.price_change_percent,
        }

    return {
        "ok": True,
        "type": "universe",
        "host": PUBLIC_REST_HOST,
        "path": TICKER_24HR_PATH,
        "authenticated": False,
        "count": len(rows),
        "watch": watch_map,
        "gainers": [_move(row) for row in gainers],
        "losers": [_move(row) for row in losers],
    }
