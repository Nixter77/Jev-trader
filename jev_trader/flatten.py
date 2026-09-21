"""Force-close every open position (MARKET reduceOnly) and sync the ledger."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from jev_trader.execution import PaperBroker, fill_price, fill_qty, is_real_fill
from jev_trader.ledger import Ledger
from jev_trader.models import CycleResult, ExecutionResult, TradeIntent
from jev_trader.status import read_json, utc_now, write_json

_LOCK = threading.RLock()


def _flatten_client_id(symbol: str) -> str:
    return f"jev1_{symbol}_flatten_{int(time.time() * 1000)}"


def _record_close(
    ledger: Ledger,
    *,
    symbol: str,
    qty: float,
    execution: ExecutionResult,
    mark: float | None,
    order_side: str,
    client_order_id: str,
) -> None:
    if not symbol or not is_real_fill(execution):
        return
    price = fill_price(execution, mark)
    intent = TradeIntent(
        action="close",
        qty=fill_qty(execution, qty),
        stop_price=None,
        stop_distance=None,
        entry_type="MARKET",
        reduce_only=True,
        client_order_id=client_order_id,
        symbol=symbol,
        risk_pct=0.0,
        order_side=order_side,
        limit_price=price,
        risk_event="flatten",
    )
    ledger.record(
        CycleResult(
            action="close",
            skip_reason=None,
            intent=intent,
            execution=execution,
            judgment=None,
            state_text="flatten",
            state={"symbol": symbol, "price": {"close": price}},
            risk_event="flatten",
        )
    )


def _flatten_ledger(broker: Any, ledger: Ledger) -> dict[str, Any]:
    closes: list[dict[str, Any]] = []
    book = ledger.book(limit=200)
    for pos in book.get("positions") or []:
        side = str(pos.get("side") or "FLAT")
        size = float(pos.get("size") or 0.0)
        symbol = str(pos.get("symbol") or "").upper()
        if not symbol or size <= 0 or side not in {"LONG", "SHORT"}:
            continue
        mark = pos.get("last_mark") or pos.get("entry")
        try:
            mark_f = None if mark is None else float(mark)
        except (TypeError, ValueError):
            mark_f = None
        cid = _flatten_client_id(symbol)
        order_side = "SELL" if side == "LONG" else "BUY"
        intent = TradeIntent(
            action="close",
            qty=size,
            stop_price=None,
            stop_distance=None,
            entry_type="MARKET",
            reduce_only=True,
            client_order_id=cid,
            symbol=symbol,
            risk_pct=0.0,
            order_side=order_side,
            limit_price=mark_f,
            risk_event="flatten",
        )
        execution = broker.submit(intent)
        _record_close(
            ledger,
            symbol=symbol,
            qty=size,
            execution=execution,
            mark=mark_f,
            order_side=order_side,
            client_order_id=cid,
        )
        closes.append(
            {
                "symbol": symbol,
                "status": execution.status,
                "detail": execution.detail,
                "qty": size,
                "side": side,
                "client_order_id": cid,
                "order_side": order_side,
            }
        )
    return {"cancelled": [], "closes": closes, "wallet": None}


def flatten_open_positions(broker: Any, ledger: Ledger) -> dict[str, Any]:
    """Cancel working orders, MARKET-close everything, then match the ledger to it."""
    with _LOCK:
        flatten = getattr(broker, "flatten_all", None)
        if callable(flatten) and not isinstance(broker, PaperBroker):
            raw = flatten() or {}
            closes = list(raw.get("closes") or [])
            for row in closes:
                if not isinstance(row, dict):
                    continue
                detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
                execution = ExecutionResult(
                    status=str(row.get("status") or "rejected"),
                    venue=str(getattr(broker, "venue", "binance_testnet")),
                    client_order_id=str(row.get("client_order_id") or ""),
                    reduce_only=True,
                    detail=detail,
                )
                _record_close(
                    ledger,
                    symbol=str(row.get("symbol") or ""),
                    qty=float(row.get("qty") or 0.0),
                    execution=execution,
                    mark=fill_price(execution, None),
                    order_side=str(row.get("order_side") or "SELL"),
                    client_order_id=str(row.get("client_order_id") or ""),
                )
            wallet = raw.get("wallet")
            if isinstance(wallet, dict):
                ledger.sync_exchange_positions(wallet)
            return {
                "cancelled": raw.get("cancelled") or [],
                "closes": closes,
                "wallet": wallet,
            }
        return _flatten_ledger(broker, ledger)


def apply_flatten_to_status(status_path: str | Path | None, result: dict[str, Any]) -> None:
    if not status_path:
        return
    status = read_json(status_path) or {}
    wallet = result.get("wallet")
    if isinstance(wallet, dict):
        status["wallet"] = wallet
    elif isinstance(status.get("wallet"), dict):
        w = dict(status["wallet"])
        w["positions"] = []
        w["open_positions"] = 0
        status["wallet"] = w
    status["ts"] = utc_now()
    status["hint"] = "принудительное закрытие с монитора"
    write_json(status_path, status)


def run_flatten(
    broker: Any,
    ledger: Ledger,
    *,
    status_path: str | Path | None = None,
    runner: Any | None = None,
) -> dict[str, Any]:
    result = flatten_open_positions(broker, ledger)
    if runner is not None:
        wallet = result.get("wallet")
        if isinstance(wallet, dict):
            runner.wallet_box["wallet"] = wallet
        try:
            runner.write_status()
        except Exception:  # noqa: BLE001 — blotter must still refresh from disk
            apply_flatten_to_status(status_path, result)
    else:
        apply_flatten_to_status(status_path, result)
    return result
