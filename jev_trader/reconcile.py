"""Exchange reconciliation, once per poll pass (rate-limited).

The bot's own cycle only sees what an order looked like right after it was
sent. A GTX (post-only) entry usually comes back NEW and fills later; an
exchange STOP_MARKET or a manual close never passes through the bot at all.
This module asks Binance what actually happened and writes it to the ledger:

1. Pending orders (resting entries, timed-out submits) are queried by
   clientOrderId. Entries still resting after one bar are cancelled. Any
   executed quantity is recorded as a fill from userTrades (price, qty,
   commission), and the stop is armed right away.
2. An entry the ledger still holds but the wallet no longer shows was closed
   on the exchange (stop / manual / liquidation): its SELL trades are recorded
   as a close with the exchange realizedPnl, commissions and funding.
3. Bot fills get the exchange's price / qty / commission, and closes get the
   exchange net PnL instead of the self-computed one.

Hourly entry cap, min-hold and the loss streak read those rows.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from jev_trader.cycle import arm_exchange_stop
from jev_trader.execution import (
    client_order_id as new_client_order_id,
    fill_price,
    fill_qty,
    is_real_fill,
    order_executed_qty,
    order_fill_price,
    order_lock,
    order_not_found,
)
from jev_trader.ledger import Ledger
from jev_trader.models import CycleResult, market_close_intent
from jev_trader.status import log_event

RESTING_STATUSES = frozenset({"NEW", "PARTIALLY_FILLED"})
STOP_WOULD_TRIGGER = -2021


def _ms_to_iso(ms: Any) -> str:
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc).isoformat(
        timespec="microseconds"
    )


def _iso_to_ms(raw: Any) -> int | None:
    try:
        ts = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)


def _f(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any] | None:
    """VWAP, qty, USDT commission, gross realizedPnl and last trade time."""
    qty = 0.0
    notional = 0.0
    commission = 0.0
    realized = 0.0
    last_ms = 0
    other_assets: set[str] = set()
    for row in trades:
        q = _f(row.get("qty"))
        px = _f(row.get("price"))
        if q <= 0 or px <= 0:
            continue
        qty += q
        notional += q * px
        asset = str(row.get("commissionAsset") or "USDT").upper()
        if asset == "USDT":
            commission += _f(row.get("commission"))
        else:
            other_assets.add(asset)
        realized += _f(row.get("realizedPnl"))
        last_ms = max(last_ms, int(_f(row.get("time"))))
    if qty <= 0:
        return None
    return {
        "qty": qty,
        "price": notional / qty,
        "commission_usdt": commission,
        "gross_pnl_usdt": realized,
        "time_ms": last_ms or None,
        "non_usdt_commission": sorted(other_assets),
    }


class Reconciler:
    def __init__(
        self,
        broker: Any,
        ledger: Ledger,
        *,
        wallet_fn: Callable[[], dict[str, Any] | None] | None = None,
        on_wallet: Callable[[dict[str, Any]], None] | None = None,
        lookback_sec: float = 72 * 3600.0,
        enrich_lookback_sec: float = 24 * 3600.0,
        entry_max_age_sec: float = 300.0,
        min_interval_sec: float = 10.0,
        max_enrich_per_pass: int = 10,
        max_pending_per_pass: int = 25,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.broker = broker
        self.ledger = ledger
        self.wallet_fn = wallet_fn
        self.on_wallet = on_wallet
        self.lookback_sec = float(lookback_sec)
        self.enrich_lookback_sec = float(enrich_lookback_sec)
        self.entry_max_age_sec = float(entry_max_age_sec)
        self.min_interval_sec = float(min_interval_sec)
        self.max_enrich_per_pass = int(max_enrich_per_pass)
        self.max_pending_per_pass = int(max_pending_per_pass)
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.venue = str(getattr(broker, "venue", "binance_testnet"))
        self._last_run = float("-inf")
        # entry client_order_id -> passes where the wallet was flat but no SELL
        # trade was found; stop asking after a few.
        self._close_misses: dict[str, int] = {}

    # --- entry point ---------------------------------------------------------

    def maybe_run(self) -> dict[str, Any] | None:
        now_mono = time.monotonic()
        if now_mono - self._last_run < self.min_interval_sec:
            return None
        self._last_run = now_mono
        return self.run_once()

    def run_once(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "entries": [],
            "closes": [],
            "cancelled": [],
            "final": [],
            "stops": [],
            "enriched": 0,
            "errors": [],
        }
        now = self.now_fn()
        # Enrich before detecting closes so an exchange close sees the entry fee.
        for step in (self._resolve_pending, self._enrich_fills, self._detect_exchange_closes):
            try:
                step(now, summary)
            except Exception as exc:  # noqa: BLE001 — never take the trading loop down
                error = f"{step.__name__}: {type(exc).__name__}: {exc}"
                summary["errors"].append(error)
                log_event("reconcile_error", error=error)
        if summary["entries"] or summary["closes"] or summary["cancelled"] or summary["stops"]:
            log_event(
                "reconcile",
                entries=summary["entries"],
                closes=summary["closes"],
                cancelled=summary["cancelled"],
                stops=summary["stops"],
                enriched=summary["enriched"],
            )
        return summary

    # --- wallet --------------------------------------------------------------

    def _fresh_wallet(self, *, cached_ok: bool = False) -> dict[str, Any] | None:
        try:
            if cached_ok:
                # Broker-side short TTL cache: fine for "is it still long?".
                wallet = self.broker.fetch_wallet()
            elif self.wallet_fn is not None:
                wallet = self.wallet_fn()
            else:
                invalidate = getattr(self.broker, "invalidate_wallet", None)
                if callable(invalidate):
                    invalidate()
                wallet = self.broker.fetch_wallet()
        except Exception as exc:  # noqa: BLE001
            log_event("reconcile_wallet_error", error=f"{type(exc).__name__}: {exc}")
            return None
        if isinstance(wallet, dict) and self.on_wallet is not None:
            try:
                self.on_wallet(wallet)
            except Exception as exc:  # noqa: BLE001
                log_event("reconcile_wallet_error", error=f"{type(exc).__name__}: {exc}")
        return wallet if isinstance(wallet, dict) else None

    @staticmethod
    def _wallet_longs(wallet: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in wallet.get("positions") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("side") or "").upper() == "LONG" and _f(row.get("size")) > 0:
                out[str(row.get("symbol") or "").upper()] = _f(row.get("size"))
        return out

    # --- 1. pending orders ---------------------------------------------------

    def _resolve_pending(self, now: datetime, summary: dict[str, Any]) -> None:
        rows = self.ledger.pending_orders(now - timedelta(seconds=self.lookback_sec))
        # Newest first: a fresh maker fill needs its stop now; old leftovers can
        # wait for the next pass.
        rows = list(reversed(rows))[: max(1, self.max_pending_per_pass)]
        armed: dict[str, dict[str, Any]] = {}
        for row in rows:
            symbol = str(row["symbol"]).upper()
            cid = str(row["client_order_id"])
            placed_ms = _iso_to_ms(row["ts"])
            age = None if placed_ms is None else now.timestamp() - placed_ms / 1000.0
            with order_lock:
                status, body = self.broker.query_order(symbol, cid)
                if order_not_found(body):
                    self.ledger.mark_order_final(int(row["id"]), "not_found")
                    summary["final"].append({"cid": cid, "status": "not_found"})
                    continue
                if not (200 <= status < 300) or not isinstance(body, dict):
                    summary["errors"].append(f"query {cid}: HTTP {status}: {body}")
                    log_event("reconcile_query_error", cid=cid, http_status=status, body=body)
                    continue
                ostatus = str(body.get("status") or "").upper()
                if ostatus in RESTING_STATUSES:
                    stale = row["action"] == "buy_long" and age is not None and age >= self.entry_max_age_sec
                    if not stale:
                        continue
                    cancel = self.broker.cancel_order(symbol, cid)
                    summary["cancelled"].append({"cid": cid, "symbol": symbol, "http_status": cancel.get("http_status")})
                    q_status, q_body = self.broker.query_order(symbol, cid)
                    if 200 <= q_status < 300 and isinstance(q_body, dict):
                        body = q_body
                        ostatus = str(body.get("status") or "").upper()
                    if ostatus in RESTING_STATUSES:
                        # Cancel did not land; try again next pass.
                        continue
                executed = order_executed_qty(body)
                if executed > 0:
                    recorded = self._record_order_fill(row, body, summary)
                    if recorded is not None and row["action"] == "buy_long":
                        # Rows go newest first: keep the newest fill per symbol.
                        armed.setdefault(symbol, recorded)
                self.ledger.mark_order_final(
                    int(row["id"]),
                    (ostatus or "unknown").lower(),
                    exchange_order_id=body.get("orderId"),
                )
                summary["final"].append({"cid": cid, "status": ostatus.lower(), "executed": executed})
        if armed:
            self._after_entry_fills(armed, summary)

    def _trades_for_order(self, symbol: str, order_id: Any) -> list[dict[str, Any]]:
        if order_id is None:
            return []
        trades_fn = getattr(self.broker, "user_trades", None)
        if not callable(trades_fn):
            return []
        try:
            return [t for t in trades_fn(symbol, order_id=order_id) if str(t.get("orderId")) == str(order_id)]
        except Exception as exc:  # noqa: BLE001
            log_event("reconcile_trades_error", symbol=symbol, order_id=order_id, error=f"{type(exc).__name__}: {exc}")
            return []

    def _record_order_fill(
        self, row: dict[str, Any], body: dict[str, Any], summary: dict[str, Any]
    ) -> dict[str, Any] | None:
        symbol = str(row["symbol"]).upper()
        cid = str(row["client_order_id"])
        order_id = body.get("orderId")
        trades = self._trades_for_order(symbol, order_id)
        agg = summarize_trades(trades)
        if agg is None:
            price = order_fill_price(body)
            if price is None:
                return None
            when = body.get("updateTime") or body.get("time")
            agg = {
                "qty": order_executed_qty(body),
                "price": price,
                "commission_usdt": None,
                "gross_pnl_usdt": None,
                "time_ms": None if when is None else int(when),
                "non_usdt_commission": [],
            }
        ts = _ms_to_iso(agg["time_ms"]) if agg.get("time_ms") else row["ts"]
        action = str(row["action"])
        gross = net = funding = None
        if action == "close":
            gross, funding, net = self._close_net(symbol, ts, agg)
        ok = self.ledger.record_exchange_fill(
            ts=ts,
            client_order_id=cid,
            exchange_order_id=order_id,
            symbol=symbol,
            action=action,
            qty=agg["qty"],
            price=agg["price"],
            venue=self.venue,
            commission_usdt=agg["commission_usdt"],
            gross_pnl_usdt=gross,
            funding_usdt=funding,
            net_pnl_usdt=net,
        )
        if not ok:
            return None
        entry = {"cid": cid, "symbol": symbol, "action": action, "qty": agg["qty"], "price": agg["price"], "ts": ts}
        (summary["entries"] if action == "buy_long" else summary["closes"]).append(entry)
        return entry

    def _after_entry_fills(self, armed: dict[str, dict[str, Any]], summary: dict[str, Any]) -> None:
        """Wallet sync + exchange stop right after a confirmed entry fill."""
        wallet = self._fresh_wallet()
        if wallet is None:
            summary["errors"].append("wallet unavailable after entry fill; stop arms next cycle")
            return
        try:
            self.ledger.sync_exchange_positions(wallet)
        except Exception as exc:  # noqa: BLE001
            log_event("ledger_sync_error", error=f"{type(exc).__name__}: {exc}")
        longs = self._wallet_longs(wallet)
        latest = {
            str(r["symbol"]).upper(): str(r["client_order_id"])
            for r in self.ledger.open_entries(datetime.now(timezone.utc) - timedelta(seconds=self.lookback_sec))
        }
        for symbol, fill in armed.items():
            if symbol not in longs or latest.get(symbol) != fill["cid"]:
                # Old backfill: the position was closed since, or a newer
                # entry owns the current long and its stop.
                continue
            stop = self.ledger.order_stop_price(fill["cid"])
            if stop is None or stop >= float(fill["price"]):
                summary["stops"].append({"symbol": symbol, "stop": stop, "armed": False, "reason": "no_valid_stop"})
                log_event("reconcile_stop_missing", symbol=symbol, cid=fill["cid"], stop=stop)
                continue
            res = arm_exchange_stop(self.broker, self.ledger, symbol, stop, None)
            http_status = res.get("http_status") if isinstance(res, dict) else None
            body = res.get("body") if isinstance(res, dict) else None
            ok = isinstance(res, dict) and not res.get("error") and (http_status is None or 200 <= int(http_status) < 300)
            row = {"symbol": symbol, "stop": stop, "armed": ok}
            if not ok:
                row["error"] = res.get("error") if isinstance(res, dict) and res.get("error") else body
                log_event("reconcile_stop_error", symbol=symbol, stop=stop, http_status=http_status, body=body)
                if isinstance(body, dict) and body.get("code") == STOP_WOULD_TRIGGER:
                    row["fail_closed"] = self._fail_closed(symbol, longs[symbol], stop)
            summary["stops"].append(row)

    def _fail_closed(self, symbol: str, size: float, stop: float) -> str:
        """Price is already through the stop: close MARKET instead of holding unprotected."""
        cid = new_client_order_id(symbol, "close")
        intent = market_close_intent(
            symbol=symbol,
            qty=size,
            order_side="SELL",
            client_order_id=cid,
            risk_event="stop",
        )
        with order_lock:
            execution = self.broker.submit(intent)
            if is_real_fill(execution):
                intent = market_close_intent(
                    symbol=symbol,
                    qty=fill_qty(execution, size),
                    order_side="SELL",
                    client_order_id=cid,
                    risk_event="stop",
                    limit_price=fill_price(execution, stop),
                )
            self.ledger.record(
                CycleResult(
                    action="close",
                    skip_reason=None,
                    intent=intent,
                    execution=execution,
                    judgment=None,
                    state_text="reconcile_stop_would_trigger",
                    state={"symbol": symbol, "price": {"close": intent.limit_price or stop}},
                    risk_event="stop",
                )
            )
        log_event("reconcile_fail_closed", symbol=symbol, stop=stop, status=execution.status)
        return execution.status

    # --- 2. closes the bot did not send ------------------------------------

    def _detect_exchange_closes(self, now: datetime, summary: dict[str, Any]) -> None:
        open_rows = [
            r for r in self.ledger.open_entries(now - timedelta(seconds=self.lookback_sec))
            if self._close_misses.get(str(r["client_order_id"]), 0) < 3
        ]
        if not open_rows:
            return
        trades_fn = getattr(self.broker, "user_trades", None)
        if not callable(trades_fn):
            return
        wallet = self._fresh_wallet(cached_ok=True)
        if wallet is None:
            return
        longs = self._wallet_longs(wallet)
        now_ms = int(now.timestamp() * 1000)
        for entry in open_rows:
            symbol = str(entry["symbol"]).upper()
            if symbol in longs:
                continue
            entry_ms = _iso_to_ms(entry["ts"])
            if entry_ms is None:
                continue
            with order_lock:
                trades = trades_fn(symbol, start_ms=entry_ms, end_ms=now_ms)
                known = self.ledger.known_order_ids(symbol, str(entry["ts"]))
                groups: dict[str, list[dict[str, Any]]] = {}
                for trade in trades:
                    if str(trade.get("side") or "").upper() != "SELL":
                        continue
                    if int(_f(trade.get("time"))) < entry_ms:
                        continue
                    oid = str(trade.get("orderId"))
                    if oid in known:
                        continue
                    groups.setdefault(oid, []).append(trade)
                if not groups:
                    key = str(entry["client_order_id"])
                    self._close_misses[key] = self._close_misses.get(key, 0) + 1
                    continue
                for oid, rows in sorted(groups.items(), key=lambda kv: max(_f(t.get("time")) for t in kv[1])):
                    agg = summarize_trades(rows)
                    if agg is None:
                        continue
                    ts = _ms_to_iso(agg["time_ms"]) if agg.get("time_ms") else now.isoformat()
                    gross, funding, net = self._close_net(symbol, ts, agg)
                    ok = self.ledger.record_exchange_fill(
                        ts=ts,
                        client_order_id=f"x{oid}",
                        exchange_order_id=oid,
                        symbol=symbol,
                        action="close",
                        qty=agg["qty"],
                        price=agg["price"],
                        venue=self.venue,
                        commission_usdt=agg["commission_usdt"],
                        gross_pnl_usdt=gross,
                        funding_usdt=funding,
                        net_pnl_usdt=net,
                    )
                    if ok:
                        summary["closes"].append(
                            {"cid": f"x{oid}", "symbol": symbol, "qty": agg["qty"], "price": agg["price"], "net_pnl_usdt": net, "ts": ts}
                        )

    # --- 3. exchange numbers for bot fills -----------------------------------

    def _close_net(
        self, symbol: str, close_ts: str, agg: dict[str, Any]
    ) -> tuple[float | None, float | None, float | None]:
        """(gross realizedPnl, funding, net). Net = gross - close fee - entry fee + funding."""
        gross = agg.get("gross_pnl_usdt")
        if gross is None:
            return None, None, None
        close_fee = float(agg.get("commission_usdt") or 0.0)
        entry = self.ledger.entry_fill_before(symbol, close_ts)
        prev_close_ms = _iso_to_ms(self.ledger.close_ts_before(symbol, close_ts) or "")
        entry_ms = None if entry is None else _iso_to_ms(entry.get("ts"))
        if entry_ms is not None and prev_close_ms is not None and prev_close_ms >= entry_ms:
            # That buy_long belongs to an earlier round trip (its fee was
            # charged on that close), or this is a second partial close.
            entry, entry_ms = None, None
        entry_fee = 0.0
        funding: float | None = None
        if entry is not None:
            entry_fee = float(entry.get("commission_usdt") or 0.0)
        # Funding since this position opened; with the entry fill missing, the
        # previous close bounds it (the book was flat in between).
        start_ms = entry_ms if entry_ms is not None else prev_close_ms
        if start_ms is not None:
            end_ms = _iso_to_ms(close_ts)
            funding_fn = getattr(self.broker, "funding_income", None)
            if callable(funding_fn) and end_ms is not None and end_ms > start_ms:
                try:
                    funding = funding_fn(symbol, start_ms, end_ms)
                except Exception as exc:  # noqa: BLE001
                    log_event("reconcile_funding_error", symbol=symbol, error=f"{type(exc).__name__}: {exc}")
        net = float(gross) - close_fee - entry_fee + float(funding or 0.0)
        return float(gross), funding, net

    def _enrich_fills(self, now: datetime, summary: dict[str, Any]) -> None:
        since = now - timedelta(seconds=self.enrich_lookback_sec)
        for fill in self.ledger.fills_to_enrich(since, limit=self.max_enrich_per_pass):
            symbol = str(fill["symbol"]).upper()
            order_id = fill.get("order_eid")
            trades = self._trades_for_order(symbol, order_id)
            agg = summarize_trades(trades)
            if agg is None:
                self.ledger.bump_enrich_tries(int(fill["id"]))
                continue
            gross = funding = net = None
            if fill["action"] == "close":
                gross, funding, net = self._close_net(symbol, str(fill["ts"]), agg)
            self.ledger.apply_exchange_enrichment(
                int(fill["id"]),
                exchange_order_id=order_id,
                price=agg["price"],
                qty=agg["qty"],
                commission_usdt=agg["commission_usdt"],
                gross_pnl_usdt=gross,
                funding_usdt=funding,
                net_pnl_usdt=net,
            )
            summary["enriched"] += 1
