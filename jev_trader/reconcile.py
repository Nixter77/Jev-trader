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

A pass has a time budget (~5 s). The first transport error / 5xx / 418 / 429
stops the pass, and the next one waits for Retry-After or an exponential
backoff. Network reads run outside `order_lock`; only ledger writes, cancels
of a resolved order and fail-closed exits take it, so a slow Binance does not
hold up flatten or cycle closes.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from jev_trader.cycle import arm_exchange_stop
from jev_trader.execution import (
    ExchangeHTTPError,
    client_order_id as new_client_order_id,
    is_backoff_status,
    retry_after_sec,
    fill_price,
    fill_qty,
    is_real_fill,
    order_executed_qty,
    order_fill_price,
    order_lock,
    order_not_found,
    wallet_positions_known,
)
from jev_trader.ledger import Ledger
from jev_trader.models import CycleResult, market_close_intent
from jev_trader.status import log_event

RESTING_STATUSES = frozenset({"NEW", "PARTIALLY_FILLED"})
STOP_WOULD_TRIGGER = -2021


class _Halt(Exception):
    """Stop this pass: the exchange is unreachable or throttling us."""

    def __init__(self, what: str, status: int, body: Any, retry_after: float | None) -> None:
        super().__init__(f"{what}: HTTP {status}")
        self.what = what
        self.status = int(status)
        self.body = body
        self.retry_after = retry_after


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
        budget_sec: float = 5.0,
        submit_unknown_grace_sec: float = 300.0,
        close_miss_reset_sec: float = 1800.0,
        min_backoff_sec: float = 10.0,
        max_backoff_sec: float = 300.0,
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
        # Resting entries with a partial fill whose closePosition stop is armed.
        self._partial_armed: set[str] = set()
        self.budget_sec = float(budget_sec)
        self.submit_unknown_grace_sec = float(submit_unknown_grace_sec)
        self.close_miss_reset_sec = float(close_miss_reset_sec)
        # cid -> monotonic time of its last close-detect miss.
        self._close_miss_at: dict[str, float] = {}
        self.min_backoff_sec = float(min_backoff_sec)
        self.max_backoff_sec = float(max_backoff_sec)
        self._deadline = float("inf")
        self._backoff = 0.0
        self._next_allowed = float("-inf")
        # entry client_order_id -> passes where the wallet was flat but no SELL
        # trade was found; stop asking after a few.
        self._close_misses: dict[str, int] = {}

    # --- entry point ---------------------------------------------------------

    def maybe_run(self) -> dict[str, Any] | None:
        now_mono = time.monotonic()
        if now_mono < self._next_allowed:
            return None
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
        self._deadline = time.monotonic() + self.budget_sec
        halted = False
        try:
            # Enrich before detecting closes so an exchange close sees the entry fee.
            for step in (self._resolve_pending, self._enrich_fills, self._detect_exchange_closes):
                if self._out_of_time(summary):
                    break
                try:
                    step(now, summary)
                except _Halt as halt:
                    self._halt(halt, summary)
                    halted = True
                    break
                except Exception as exc:  # noqa: BLE001 — never take the trading loop down
                    error = f"{step.__name__}: {type(exc).__name__}: {exc}"
                    summary["errors"].append(error)
                    log_event("reconcile_error", error=error)
        finally:
            self._deadline = float("inf")
        if not halted:
            self._backoff = 0.0
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

    # --- budget / backoff ----------------------------------------------------

    def _time_left(self) -> float:
        return self._deadline - time.monotonic()

    def _out_of_time(self, summary: dict[str, Any]) -> bool:
        if self._time_left() > 0:
            return False
        summary["budget_exhausted"] = True
        return True

    def _timeout_kw(self, fn: Any) -> dict[str, float]:
        """Per-request timeout bounded by what is left of the pass budget."""
        try:
            accepts = "timeout" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            accepts = False
        if not accepts:
            return {}
        return {"timeout": max(1.0, min(10.0, self._time_left()))}

    @staticmethod
    def _check(what: str, status: Any, body: Any) -> None:
        try:
            code = int(status)
        except (TypeError, ValueError):
            return
        if is_backoff_status(code):
            raise _Halt(what, code, body, retry_after_sec(body))

    @staticmethod
    def _halt_from(what: str, exc: ExchangeHTTPError) -> _Halt:
        return _Halt(what, exc.status, exc.body, exc.retry_after)

    def _halt(self, halt: _Halt, summary: dict[str, Any]) -> None:
        self._backoff = min(
            self.max_backoff_sec, max(self.min_backoff_sec, self._backoff * 2.0)
        )
        wait = max(self._backoff, float(halt.retry_after or 0.0))
        self._next_allowed = time.monotonic() + wait
        summary["halted"] = {"what": halt.what, "http_status": halt.status, "wait_sec": wait}
        summary["errors"].append(f"halted {halt.what}: HTTP {halt.status}; next pass in {wait:.0f}s")
        log_event(
            "reconcile_halt",
            what=halt.what,
            http_status=halt.status,
            retry_after=halt.retry_after,
            wait_sec=wait,
        )

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
        if isinstance(wallet, dict) and not wallet_positions_known(wallet):
            # Balance fallback: no positions list, so "not long" means nothing.
            log_event("reconcile_wallet_positions_unknown")
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
        halted: _Halt | None = None
        for row in rows:
            if self._out_of_time(summary):
                break
            try:
                self._resolve_one(row, now, summary, armed)
            except _Halt as halt:
                halted = halt
                break
        # Fills already written are final (not re-queried), so this is their
        # only chance at the intended stop before the next bar. A halt on some
        # other row does not change that; only an IP ban (418) does, where
        # every extra request lengthens the ban.
        if armed and (halted is None or halted.status != 418):
            self._after_entry_fills(armed, summary)
        if halted is not None:
            raise halted

    def _resolve_one(
        self,
        row: dict[str, Any],
        now: datetime,
        summary: dict[str, Any],
        armed: dict[str, dict[str, Any]],
    ) -> None:
        symbol = str(row["symbol"]).upper()
        cid = str(row["client_order_id"])
        placed_ms = _iso_to_ms(row["ts"])
        age = None if placed_ms is None else now.timestamp() - placed_ms / 1000.0
        query = self.broker.query_order
        status, body = query(symbol, cid, **self._timeout_kw(query))
        if order_not_found(body):
            if str(row.get("status") or "") != "submit_unknown":
                with order_lock:
                    self.ledger.mark_order_final(int(row["id"]), "not_found")
                summary["final"].append({"cid": cid, "status": "not_found"})
                return
            if age is None or age < self.submit_unknown_grace_sec:
                # Submit timed out / 5xx: -2013 may just be exchange lag.
                summary.setdefault("unknown_waiting", []).append({"cid": cid, "age_sec": age})
                return
            # Grace over: cancel by cid in case it surfaces, then settle.
            cancel = self.broker.cancel_order(symbol, cid, **self._timeout_kw(self.broker.cancel_order))
            c_status, c_body = cancel.get("http_status"), cancel.get("body")
            self._check("cancel_order", c_status, c_body)
            if (
                isinstance(c_status, int)
                and 200 <= c_status < 300
                and isinstance(c_body, dict)
                and c_body.get("orderId") is not None
            ):
                summary["cancelled"].append({"cid": cid, "symbol": symbol, "http_status": c_status})
                body = c_body
                status = c_status
            else:
                with order_lock:
                    self.ledger.mark_order_final(int(row["id"]), "not_found")
                summary["final"].append({"cid": cid, "status": "not_found", "after_grace": True})
                log_event("reconcile_submit_unknown_settled", cid=cid, symbol=symbol, age_sec=age)
                return
        self._check("query_order", status, body)
        if not (200 <= status < 300) or not isinstance(body, dict):
            summary["errors"].append(f"query {cid}: HTTP {status}: {body}")
            log_event("reconcile_query_error", cid=cid, http_status=status, body=body)
            return
        ostatus = str(body.get("status") or "").upper()
        if ostatus in RESTING_STATUSES:
            stale = row["action"] == "buy_long" and age is not None and age >= self.entry_max_age_sec
            if not stale and row["action"] == "buy_long" and order_executed_qty(body) > 0:
                # Part of the entry is already a position: protect it now, not
                # when the order finishes. Price already through the stop:
                # cancel the rest and settle it like a stale order below.
                if self._arm_partial(symbol, cid, body, summary) == "would_trigger":
                    stale = True
            if not stale:
                return
            cancel = self.broker.cancel_order(symbol, cid, **self._timeout_kw(self.broker.cancel_order))
            summary["cancelled"].append({"cid": cid, "symbol": symbol, "http_status": cancel.get("http_status")})
            self._check("cancel_order", cancel.get("http_status"), cancel.get("body"))
            q_status, q_body = query(symbol, cid, **self._timeout_kw(query))
            self._check("query_order", q_status, q_body)
            if 200 <= q_status < 300 and isinstance(q_body, dict):
                body = q_body
                ostatus = str(body.get("status") or "").upper()
            if ostatus in RESTING_STATUSES:
                # Cancel did not land; try again next pass.
                return
        executed = order_executed_qty(body)
        if executed > 0:
            recorded = self._record_order_fill(row, body, summary)
            if recorded is not None and row["action"] == "buy_long":
                # Rows go newest first: keep the newest fill per symbol.
                armed.setdefault(symbol, recorded)
        with order_lock:
            self.ledger.mark_order_final(
                int(row["id"]),
                (ostatus or "unknown").lower(),
                exchange_order_id=body.get("orderId"),
            )
        self._partial_armed.discard(cid)
        summary["final"].append({"cid": cid, "status": ostatus.lower(), "executed": executed})

    def _arm_partial(self, symbol: str, cid: str, body: dict[str, Any], summary: dict[str, Any]) -> str:
        """closePosition stop for a still-resting entry with executedQty > 0."""
        if cid in self._partial_armed:
            return "armed"
        executed = order_executed_qty(body)
        stop = self.ledger.order_stop_price(cid)
        price = _f(body.get("price")) or _f(body.get("avgPrice"))
        if stop is None or stop <= 0 or (price > 0 and stop >= price):
            self._partial_armed.add(cid)  # log once; the final fill path retries
            summary["stops"].append({"symbol": symbol, "stop": stop, "armed": False, "reason": "no_valid_stop", "partial": True})
            log_event("reconcile_stop_missing", symbol=symbol, cid=cid, stop=stop, partial=True)
            return "skip"
        res = arm_exchange_stop(self.broker, self.ledger, symbol, stop, None)
        http_status = res.get("http_status") if isinstance(res, dict) else None
        res_body = res.get("body") if isinstance(res, dict) else None
        ok = isinstance(res, dict) and not res.get("error") and (http_status is None or 200 <= int(http_status) < 300)
        row = {"symbol": symbol, "stop": stop, "armed": ok, "partial": True, "executed": executed}
        summary["stops"].append(row)
        if ok:
            self._partial_armed.add(cid)
            log_event("reconcile_partial_stop", symbol=symbol, cid=cid, stop=stop, executed=executed)
            return "armed"
        row["error"] = res.get("error") if isinstance(res, dict) and res.get("error") else res_body
        log_event("reconcile_stop_error", symbol=symbol, stop=stop, http_status=http_status, body=res_body, partial=True)
        self._check("stop", http_status, res_body)
        if isinstance(res_body, dict) and res_body.get("code") == STOP_WOULD_TRIGGER:
            row["fail_closed"] = "cancel_rest_then_close"
            return "would_trigger"
        return "error"

    def _trades_for_order(self, symbol: str, order_id: Any) -> list[dict[str, Any]]:
        if order_id is None:
            return []
        trades_fn = getattr(self.broker, "user_trades", None)
        if not callable(trades_fn):
            return []
        try:
            rows = trades_fn(symbol, order_id=order_id, **self._timeout_kw(trades_fn))
            return [t for t in rows if str(t.get("orderId")) == str(order_id)]
        except ExchangeHTTPError as exc:
            if exc.backoff:
                raise self._halt_from("user_trades", exc) from exc
            log_event("reconcile_trades_error", symbol=symbol, order_id=order_id, error=str(exc))
            return []
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
        with order_lock:
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
                fee_unaccounted=self._fee_unaccounted(symbol, cid, agg),
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
        self._expire_close_misses()
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
            if self._out_of_time(summary):
                return
            try:
                trades = trades_fn(symbol, start_ms=entry_ms, end_ms=now_ms, **self._timeout_kw(trades_fn))
            except ExchangeHTTPError as exc:
                if exc.backoff:
                    raise self._halt_from("user_trades", exc) from exc
                raise
            groups = self._unknown_sell_groups(symbol, str(entry["ts"]), entry_ms, trades)
            if not groups:
                key = str(entry["client_order_id"])
                self._close_misses[key] = self._close_misses.get(key, 0) + 1
                self._close_miss_at[key] = time.monotonic()
                continue
            for oid, rows in sorted(groups.items(), key=lambda kv: max(_f(t.get("time")) for t in kv[1])):
                agg = summarize_trades(rows)
                if agg is None:
                    continue
                ts = _ms_to_iso(agg["time_ms"]) if agg.get("time_ms") else now.isoformat()
                # Funding lookup is network: outside the lock.
                gross, funding, net = self._close_net(symbol, ts, agg)
                with order_lock:
                    # A flatten may have recorded this order while we fetched.
                    if oid in self.ledger.known_order_ids(symbol, str(entry["ts"])):
                        continue
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
                        fee_unaccounted=self._fee_unaccounted(symbol, f"x{oid}", agg),
                    )
                if ok:
                    summary["closes"].append(
                        {"cid": f"x{oid}", "symbol": symbol, "qty": agg["qty"], "price": agg["price"], "net_pnl_usdt": net, "ts": ts}
                    )

    def _expire_close_misses(self) -> None:
        """Give a gave-up entry another 3 tries every close_miss_reset_sec."""
        now_mono = time.monotonic()
        for key in [k for k, n in self._close_misses.items() if n >= 3]:
            if now_mono - self._close_miss_at.get(key, now_mono) >= self.close_miss_reset_sec:
                self._close_misses.pop(key, None)
                self._close_miss_at.pop(key, None)

    def _unknown_sell_groups(
        self, symbol: str, since_ts: str, entry_ms: int, trades: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """SELL trades after the entry, by orderId, that no ledger fill owns yet."""
        known = self.ledger.known_order_ids(symbol, since_ts)
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
        return groups

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
            if entry.get("fee_unaccounted"):
                agg["entry_fee_unaccounted"] = True
        # Funding since this position opened; with the entry fill missing, the
        # previous close bounds it (the book was flat in between).
        start_ms = entry_ms if entry_ms is not None else prev_close_ms
        if start_ms is not None:
            end_ms = _iso_to_ms(close_ts)
            funding_fn = getattr(self.broker, "funding_income", None)
            if callable(funding_fn) and end_ms is not None and end_ms > start_ms:
                # One retry on a soft failure (None / non-throttle error);
                # a throttle or transport error halts the pass instead.
                for attempt in range(2):
                    try:
                        funding = funding_fn(symbol, start_ms, end_ms, **self._timeout_kw(funding_fn))
                    except ExchangeHTTPError as exc:
                        if exc.backoff:
                            raise self._halt_from("funding_income", exc) from exc
                        log_event("reconcile_funding_error", symbol=symbol, error=str(exc), attempt=attempt)
                    except Exception as exc:  # noqa: BLE001
                        log_event(
                            "reconcile_funding_error",
                            symbol=symbol,
                            error=f"{type(exc).__name__}: {exc}",
                            attempt=attempt,
                        )
                    if funding is not None:
                        break
        net = float(gross) - close_fee - entry_fee + float(funding or 0.0)
        return float(gross), funding, net

    @staticmethod
    def _fee_unaccounted(symbol: str, cid: Any, agg: dict[str, Any]) -> bool:
        """Non-USDT commission (BNB) on this fill, or on the entry a close nets against."""
        assets = list(agg.get("non_usdt_commission") or [])
        if assets:
            log_event("fee_unaccounted", symbol=symbol, cid=cid, assets=assets)
        return bool(assets) or bool(agg.get("entry_fee_unaccounted"))

    def _enrich_fills(self, now: datetime, summary: dict[str, Any]) -> None:
        since = now - timedelta(seconds=self.enrich_lookback_sec)
        for fill in self.ledger.fills_to_enrich(since, limit=self.max_enrich_per_pass):
            if self._out_of_time(summary):
                return
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
                fee_unaccounted=self._fee_unaccounted(symbol, fill["client_order_id"], agg),
            )
            summary["enriched"] += 1
