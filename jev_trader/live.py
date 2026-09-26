"""5m-close live loop: public market snapshot → existing run_once.

The full UM ticker list is scanned. Jev runs once per closed 5m bar on the
liquid watch list (plus open positions), not on every tick and not on all 700+
symbols.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from jev_trader.cycle import decision_payload, dumps_decision, run_once
from jev_trader.execution import (
    BinanceFuturesBroker,
    BinanceTestnetBroker,
    PaperBroker,
    is_real_fill,
    wallet_positions_known,
)
from jev_trader.ledger import Ledger
from jev_trader.risk import (
    build_entry_guard_state,
    load_entry_guard_config,
    loss_streak_from_closes,
)
from jev_trader.status import (
    log_event,
    resolve_day_anchor,
    seconds_to_next_5m,
    utc_now,
    write_json,
)
from jev_trader.models import (
    AccountState,
    Candle,
    CycleResult,
    JevJudgment,
    MarketSnapshot,
    OrderBook,
    Position,
)
from jev_trader.public_market import (
    DEFAULT_KLINE_LIMIT,
    ParsedKline,
    UniverseTicker,
    fetch_depth,
    fetch_klines,
    fetch_universe_summary,
    parse_depth,
    parse_rest_klines,
    parse_ticker_24hr,
    parse_ws_kline_event,
    rest_closed_klines,
    universe_payload,
)
from jev_trader.universe import (
    DEFAULT_MIN_QUOTE_VOLUME,
    DEFAULT_UNIVERSE_SIZE,
    order_symbols_open_first,
    select_trade_universe,
)
from jev_trader.telegram import TelegramNotifier

RunCycle = Callable[[MarketSnapshot], CycleResult]


class CandleBuffer:
    def __init__(self, max_bars: int = 300) -> None:
        self.max_bars = max_bars
        self._bars: dict[int, Candle] = {}

    def seed(self, candles: list[Candle] | tuple[Candle, ...]) -> None:
        for candle in candles:
            self._bars[candle.ts] = candle
        self._trim()

    def upsert(self, candle: Candle) -> None:
        self._bars[candle.ts] = candle
        self._trim()

    def _trim(self) -> None:
        if len(self._bars) <= self.max_bars:
            return
        for ts in sorted(self._bars)[: -self.max_bars]:
            del self._bars[ts]

    def as_tuple(self) -> tuple[Candle, ...]:
        return tuple(self._bars[ts] for ts in sorted(self._bars))


def snapshot_from_closed_bars(
    symbol: str,
    candles: tuple[Candle, ...],
    *,
    tf: str = "5m",
    book: OrderBook | None = None,
    position: Position | None = None,
    news: tuple[str, ...] = (),
    funding: float | None = None,
    doi_1h: float | None = None,
    btc_corr: float | None = None,
) -> MarketSnapshot:
    pos = position or Position(side="FLAT", size=0.0, cash_usdt=10_000.0)
    return MarketSnapshot(
        symbol=symbol,
        tf=tf,
        candles=candles,
        position=pos,
        book=book,
        news=news,
        funding=funding,
        doi_1h=doi_1h,
        btc_corr=btc_corr,
    )


class FiveMinuteCloseLoop:
    """Fire `run_cycle` once per newly closed 5m bar, not per intermediate tick."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        *,
        tf: str = "5m",
        run_cycle: RunCycle | None = None,
        position: Position | None = None,
        book: OrderBook | None = None,
        news: tuple[str, ...] = (),
        max_bars: int = 300,
    ) -> None:
        self.symbol = symbol.upper()
        self.tf = tf
        self.run_cycle = run_cycle
        self.position = position or Position(side="FLAT", size=0.0, cash_usdt=10_000.0)
        self.book = book
        self.news = news
        self.buffer = CandleBuffer(max_bars=max_bars)
        self.ledger = None
        self.last_fired_open_ts: int | None = None
        self.skip_existing_close = False
        self.invocations = 0
        self.last_snapshot: MarketSnapshot | None = None
        self.last_result: CycleResult | None = None

    def seed_klines(self, rows: list[ParsedKline] | list[Candle]) -> None:
        candles: list[Candle] = []
        for row in rows:
            candles.append(row.candle if isinstance(row, ParsedKline) else row)
        self.buffer.seed(candles)

    def handle_ws_event(self, payload: Mapping[str, Any]) -> CycleResult | None:
        parsed = parse_ws_kline_event(dict(payload))
        if parsed is None:
            return None
        symbol = (parsed.symbol or self.symbol).upper()
        if symbol != self.symbol:
            return None
        self.buffer.upsert(parsed.candle)
        if not parsed.closed:
            return None
        return self._fire_if_new(parsed.candle.ts)

    def handle_rest_klines(
        self,
        payload: Any,
        *,
        now_ms: int | None = None,
    ) -> CycleResult | None:
        rows = payload if isinstance(payload, list) and payload and isinstance(payload[0], ParsedKline) else parse_rest_klines(payload)
        closed = rest_closed_klines(rows, now_ms=now_ms)
        for row in closed:
            self.buffer.upsert(row.candle)
        if not closed:
            return None
        latest_open = closed[-1].candle.ts
        if self.skip_existing_close and self.last_fired_open_ts is None:
            self.last_fired_open_ts = latest_open
            return None
        return self._fire_if_new(latest_open)

    def _fire_if_new(self, open_ts: int) -> CycleResult | None:
        if self.last_fired_open_ts == open_ts:
            return None
        candles = self.buffer.as_tuple()
        if not candles:
            return None
        snapshot = snapshot_from_closed_bars(
            self.symbol,
            candles,
            tf=self.tf,
            book=self.book,
            position=self.position,
            news=self.news,
        )
        self.last_fired_open_ts = open_ts
        self.invocations += 1
        self.last_snapshot = snapshot
        if self.run_cycle is None:
            return None
        result = self.run_cycle(snapshot)
        self.last_result = result
        if self.ledger is not None:
            self.position = self.ledger.load_position(
                self.symbol, default_cash=self.position.cash_usdt
            )
        else:
            self._apply_fill(result)
        return result

    def _apply_fill(self, result: CycleResult) -> None:
        execution = result.execution
        intent = result.intent
        if execution is None or intent is None or intent.qty <= 0:
            return
        if not is_real_fill(execution):
            return
        cash = self.position.cash_usdt
        if result.action == "buy_long":
            self.position = Position(
                side="LONG",
                size=intent.qty,
                cash_usdt=cash,
                entry=intent.limit_price,
                bars_in_trade=0,
                stop_price=intent.stop_price,
            )
        elif result.action == "sell_short":
            self.position = Position(
                side="SHORT",
                size=intent.qty,
                cash_usdt=cash,
                entry=intent.limit_price,
                bars_in_trade=0,
            )
        elif result.action == "close":
            self.position = Position(side="FLAT", size=0.0, cash_usdt=cash)


def overlay_wallet_position(snapshot: MarketSnapshot, wallet: dict[str, Any]) -> MarketSnapshot:
    """Exchange positions are the source of truth on testnet/live."""
    cash = float(wallet.get("equity_usdt") or snapshot.position.cash_usdt)
    match: dict[str, Any] | None = None
    for row in wallet.get("positions") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol") or "").upper() == snapshot.symbol.upper():
            match = row
            break
    if match is None:
        return replace(
            snapshot,
            position=replace(
                snapshot.position,
                side="FLAT",
                size=0.0,
                entry=None,
                cash_usdt=cash,
                stop_price=None,
            ),
        )
    side = str(match.get("side") or "FLAT")
    if side not in {"FLAT", "LONG", "SHORT"}:
        side = "FLAT"
    stop = snapshot.position.stop_price if side == "LONG" else None
    return replace(
        snapshot,
        position=replace(
            snapshot.position,
            side=side,  # type: ignore[arg-type]
            size=float(match.get("size") or 0.0),
            entry=None if not match.get("entry") else float(match["entry"]),
            cash_usdt=cash,
            stop_price=stop,
        ),
    )


def _box_lock(box: dict[str, Any]) -> threading.Lock:
    # threading.Lock is a factory, not a type. setdefault keeps one lock if two threads arrive together.
    lock = box.get("_lock")
    if lock is not None:
        return lock
    return box.setdefault("_lock", threading.Lock())


def begin_wallet_pull(box: dict[str, Any]) -> int:
    """Generation observed before a slow exchange fetch."""
    with _box_lock(box):
        return int(box.get("_wallet_gen") or 0)


def finish_wallet_pull(
    box: dict[str, Any],
    gen: int,
    wallet: dict[str, Any] | None,
    *,
    error: str | None = None,
) -> dict[str, Any] | None:
    """Publish `wallet` only if nobody force-published a newer snapshot while we fetched.

    Returns the snapshot the caller should trade on. After a flatten that is the
    flat wallet, not the positions this fetch observed before the close.
    """
    with _box_lock(box):
        if int(box.get("_wallet_gen") or 0) != gen:
            current = box.get("wallet")
            return current if isinstance(current, dict) else None
        if error is not None:
            box["wallet_error"] = error
            return None
        box["wallet"] = wallet
        box.pop("wallet_error", None)
        return wallet


def force_wallet(box: dict[str, Any], wallet: dict[str, Any]) -> None:
    """Flatten (HTTP thread) wins over an in-flight fetch on the bot thread."""
    with _box_lock(box):
        box["_wallet_gen"] = int(box.get("_wallet_gen") or 0) + 1
        box["wallet"] = wallet
        box.pop("wallet_error", None)


def wallet_view(box: dict[str, Any]) -> tuple[Any, Any, Any]:
    with _box_lock(box):
        return box.get("wallet"), box.get("start_equity_usdt"), box.get("wallet_error")


def _pull_wallet(broker: Any, wallet_box: dict[str, Any] | None) -> dict[str, Any] | None:
    fetch = getattr(broker, "fetch_wallet", None)
    if not callable(fetch):
        return None
    gen = 0 if wallet_box is None else begin_wallet_pull(wallet_box)
    try:
        wallet = fetch()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        log_event("wallet_error", error=error)
        if wallet_box is None:
            return None
        return finish_wallet_pull(wallet_box, gen, None, error=error)
    if wallet_box is None:
        return wallet
    return finish_wallet_pull(wallet_box, gen, wallet)


def _account_for_cycle(
    snapshot: MarketSnapshot,
    wallet: dict[str, Any] | None,
    *,
    ledger: Ledger | None,
    wallet_box: dict[str, Any] | None,
    account_kwargs: dict[str, Any],
    max_positions: int,
    wallet_expected: bool = False,
) -> tuple[MarketSnapshot, AccountState]:
    """Wallet is the position. Ledger only supplies a stop the exchange row does not carry.

    wallet_expected: the broker has a wallet endpoint. A missing wallet then
    means "unknown", and AccountState.wallet_ok is False (no new entries).
    """
    cash = snapshot.position.cash_usdt
    available = None
    daily_pnl_pct = float(account_kwargs.get("daily_pnl_pct", 0.0))
    # Balance-only wallet: equity is real, positions are unknown (not flat).
    positions_known = wallet is not None and wallet_positions_known(wallet)
    if wallet is not None:
        if positions_known:
            snapshot = overlay_wallet_position(snapshot, wallet)
        else:
            log_event("wallet_positions_unknown", symbol=snapshot.symbol)
        cash = float(wallet.get("equity_usdt") or cash)
        if wallet.get("available_usdt") is not None:
            available = float(wallet["available_usdt"])
        if wallet_box is not None and cash:
            anchor_path = wallet_box.get("risk_anchor_path")
            venue = wallet_box.get("venue")
            if anchor_path and venue:
                # A process that stays up past UTC midnight must roll the day,
                # otherwise yesterday's loss keeps blocking entries.
                start = resolve_day_anchor(anchor_path, venue=str(venue), equity=float(cash))
                with _box_lock(wallet_box):
                    wallet_box["start_equity_usdt"] = start
            else:
                with _box_lock(wallet_box):
                    start = wallet_box.get("start_equity_usdt")
                    if not start:
                        wallet_box["start_equity_usdt"] = cash
                        start = cash
            if start:
                daily_pnl_pct = (cash - float(start)) / float(start)
        if ledger is not None and positions_known:
            try:
                ledger.sync_exchange_positions(wallet)
                if snapshot.position.stop_price is None and snapshot.position.side == "LONG":
                    loaded = ledger.load_position(snapshot.symbol, default_cash=cash)
                    if loaded.stop_price is not None:
                        snapshot = replace(
                            snapshot,
                            position=replace(snapshot.position, stop_price=loaded.stop_price),
                        )
                if wallet_box is not None:
                    with _box_lock(wallet_box):
                        wallet_box.pop("sync_error", None)
            except Exception as exc:  # noqa: BLE001 — trading must not die on ledger sync
                error = f"{type(exc).__name__}: {exc}"
                log_event("ledger_sync_error", symbol=snapshot.symbol, error=error)
                if wallet_box is not None:
                    with _box_lock(wallet_box):
                        wallet_box["sync_error"] = error
    if ledger is not None:
        open_n = ledger.count_open_positions()
    else:
        open_n = 0 if snapshot.position.side == "FLAT" else 1
    if positions_known and wallet is not None and wallet.get("open_positions") is not None:
        open_n = int(wallet["open_positions"])
    account = AccountState(
        equity_usdt=cash,
        daily_pnl_pct=daily_pnl_pct,
        kill_switch=bool(account_kwargs.get("kill_switch", False)),
        open_positions=open_n,
        max_positions=int(account_kwargs.get("max_positions", max_positions)),
        available_usdt=available,
        wallet_ok=not (wallet_expected and not positions_known),
    )
    return snapshot, account


def make_run_cycle(
    *,
    judgment: JevJudgment | None,
    account_kwargs: dict[str, Any],
    broker: PaperBroker | BinanceTestnetBroker | BinanceFuturesBroker,
    ledger: Ledger | None,
    notifier: TelegramNotifier | None,
    typesafe_api_key: str | None,
    jev_client: Any | None = None,
    follow_jev: bool = False,
    min_should_trade: float | None = None,
    max_positions: int = 3,
    wallet_box: dict[str, Any] | None = None,
) -> RunCycle:
    def run_cycle(snapshot: MarketSnapshot) -> CycleResult:
        wallet = _pull_wallet(broker, wallet_box)
        snapshot, account = _account_for_cycle(
            snapshot,
            wallet,
            ledger=ledger,
            wallet_box=wallet_box,
            account_kwargs=account_kwargs,
            max_positions=max_positions,
            wallet_expected=callable(getattr(broker, "fetch_wallet", None)),
        )
        return run_once(
            snapshot,
            judgment=judgment,
            account=account,
            broker=broker,
            ledger=ledger,
            notifier=notifier,
            typesafe_api_key=typesafe_api_key if judgment is None and jev_client is None else None,
            jev_client=jev_client if judgment is None else None,
            follow_jev=follow_jev,
            min_should_trade=min_should_trade,
        )

    return run_cycle


def _kline_payload_for_symbol(payload: Any, symbol: str) -> Any:
    if isinstance(payload, dict) and symbol in payload and isinstance(payload[symbol], list):
        return payload[symbol]
    return payload


class LiveRunner:
    def __init__(
        self,
        symbols: list[str],
        *,
        run_cycle: RunCycle,
        interval: str = "5m",
        kline_poll: float = 5.0,
        summary_interval: float = 15.0,
        max_cycles: int | None = None,
        max_runtime: float | None = None,
        print_universe: bool = False,
        kline_limit: int = DEFAULT_KLINE_LIMIT,
        recorded_klines: Any = None,
        recorded_closes: list[dict[str, Any]] | None = None,
        recorded_depth: dict[str, Any] | None = None,
        now_ms: int | None = None,
        fire_latest: bool = False,
        heartbeat: bool = True,
        ledger: Ledger | None = None,
        universe: bool = False,
        universe_size: int = DEFAULT_UNIVERSE_SIZE,
        min_quote_volume: float = DEFAULT_MIN_QUOTE_VOLUME,
        recorded_tickers: Any = None,
        status_path: str | Path | None = None,
        venue: str = "paper",
        follow_jev: bool = False,
        wallet_box: dict[str, Any] | None = None,
        decision_backend: str = "jev",
        laya_checkpoint: str = "typed-decisions",
        reconciler: Any = None,
    ) -> None:
        self.run_cycle = run_cycle
        self.reconciler = reconciler
        self.last_reconcile: dict[str, Any] | None = None
        self.symbols = [s.upper() for s in symbols]
        self.interval = interval
        self.kline_poll = max(0.0, float(kline_poll))
        self.summary_interval = max(1.0, float(summary_interval))
        self.max_cycles = max_cycles
        self.max_runtime = max_runtime
        self.print_universe = print_universe
        self.kline_limit = kline_limit
        self.recorded_klines = recorded_klines
        self.recorded_closes = list(recorded_closes or [])
        self._offline = recorded_klines is not None or bool(self.recorded_closes)
        self.now_ms = now_ms
        self.ledger = ledger
        self.universe = bool(universe)
        self.universe_size = int(universe_size)
        self.min_quote_volume = float(min_quote_volume)
        self.recorded_tickers = recorded_tickers
        self.fire_latest = bool(fire_latest)
        self.status_path = None if status_path is None else Path(status_path)
        self.venue = venue
        self.follow_jev = bool(follow_jev)
        self.wallet_box = wallet_box if wallet_box is not None else {}
        self._state_lock = threading.Lock()
        self.decision_backend = (decision_backend or "jev").strip().lower() or "jev"
        self.laya_checkpoint = (laya_checkpoint or "typed-decisions").strip() or "typed-decisions"
        self.last_decisions: list[dict[str, Any]] = []
        self.last_error: str | None = None
        self.cycles = 0
        self.loops = {
            symbol: FiveMinuteCloseLoop(symbol, tf=interval, run_cycle=run_cycle)
            for symbol in self.symbols
        }
        if ledger is not None:
            for symbol, loop in self.loops.items():
                loop.ledger = ledger
                loop.position = ledger.load_position(symbol, default_cash=loop.position.cash_usdt)
        if recorded_depth:
            book = parse_depth(recorded_depth)
            for loop in self.loops.values():
                loop.book = book
        skip_existing = not self._offline and self.max_cycles is None and not fire_latest
        self.skip_existing = skip_existing
        self.heartbeat = heartbeat and not self._offline
        for loop in self.loops.values():
            loop.skip_existing_close = skip_existing
        if recorded_klines is not None and self.recorded_closes:
            for symbol, loop in self.loops.items():
                rows = parse_rest_klines(_kline_payload_for_symbol(recorded_klines, symbol))
                loop.seed_klines(rest_closed_klines(rows, now_ms=now_ms))

    @property
    def offline(self) -> bool:
        return self._offline

    def _ensure_loop(self, symbol: str) -> FiveMinuteCloseLoop:
        symbol = symbol.upper()
        loop = self.loops.get(symbol)
        if loop is not None:
            return loop
        loop = FiveMinuteCloseLoop(symbol, tf=self.interval, run_cycle=self.run_cycle)
        loop.skip_existing_close = self.skip_existing
        if self.ledger is not None:
            loop.ledger = self.ledger
            loop.position = self.ledger.load_position(symbol, default_cash=loop.position.cash_usdt)
        self.loops[symbol] = loop
        return loop

    def _ticker_rows(self) -> list[UniverseTicker]:
        if self.recorded_tickers is not None:
            if isinstance(self.recorded_tickers, list) and self.recorded_tickers and isinstance(
                self.recorded_tickers[0], UniverseTicker
            ):
                return list(self.recorded_tickers)
            return parse_ticker_24hr(self.recorded_tickers)
        return fetch_universe_summary()

    def sync_universe(self) -> list[str]:
        extra: list[str] = []
        if self.ledger is not None:
            extra.extend(self.ledger.open_symbols())
        rows = self._ticker_rows()
        selected = select_trade_universe(
            rows,
            limit=self.universe_size,
            min_quote_volume=self.min_quote_volume,
            extra=extra,
        )
        if not selected:
            selected = list(self.symbols) or ["BTCUSDT"]
        self.symbols = selected
        for symbol in selected:
            self._ensure_loop(symbol)
        return selected

    def _open_priority_symbols(self) -> list[str]:
        """Ledger + exchange wallet opens, then in-memory loop positions."""
        ordered: list[str] = []
        seen: set[str] = set()

        def add(symbol: str) -> None:
            s = str(symbol or "").upper()
            if not s or s in seen:
                return
            ordered.append(s)
            seen.add(s)

        if self.ledger is not None:
            try:
                for symbol in self.ledger.open_symbols():
                    add(symbol)
            except Exception as exc:  # noqa: BLE001 — ordering must not break the poll
                log_event("ledger_open_symbols_error", error=f"{type(exc).__name__}: {exc}")
        wallet = self.wallet_box.get("wallet") if isinstance(self.wallet_box, dict) else None
        if isinstance(wallet, dict):
            for row in wallet.get("positions") or []:
                if not isinstance(row, dict):
                    continue
                side = str(row.get("side") or "FLAT").upper()
                size = float(row.get("size") or 0.0)
                if side in {"LONG", "SHORT"} and size > 0:
                    add(str(row.get("symbol") or ""))
        for symbol, loop in self.loops.items():
            pos = loop.position
            if pos.side in {"LONG", "SHORT"} and float(pos.size or 0.0) > 0:
                add(symbol)
        return ordered

    def _symbols_for_pass(self) -> list[str]:
        """Open positions first each poll so management is not starved by Laya latency."""
        return order_symbols_open_first(self.symbols, self._open_priority_symbols())

    def reconcile(self) -> None:
        """Exchange reconciliation (rate-limited inside the reconciler)."""
        rec = self.reconciler
        if rec is None or self._offline:
            return
        try:
            summary = rec.maybe_run()
        except Exception as exc:  # noqa: BLE001 — reconciliation must not stop the loop
            log_event("reconcile_error", error=f"{type(exc).__name__}: {exc}")
            return
        if summary is None:
            return
        with self._state_lock:
            self.last_reconcile = {
                "ts": utc_now(),
                "entries": len(summary.get("entries") or []),
                "closes": len(summary.get("closes") or []),
                "cancelled": len(summary.get("cancelled") or []),
                "stops": len(summary.get("stops") or []),
                "enriched": int(summary.get("enriched") or 0),
                "errors": list(summary.get("errors") or [])[:5],
            }

    def poll_live_symbol(self, symbol: str) -> CycleResult | None:
        loop = self._ensure_loop(symbol)
        rows = fetch_klines(symbol, interval=self.interval, limit=self.kline_limit)
        try:
            loop.book = fetch_depth(symbol)
        except Exception:  # noqa: BLE001 — book is optional for a decision cycle
            pass
        return loop.handle_rest_klines(rows, now_ms=self.now_ms)

    def _record_result(self, result: CycleResult) -> None:
        print(dumps_decision(result), flush=True)
        compact = self._compact_decision(result)
        with self._state_lock:
            self.last_decisions.insert(0, compact)
            del self.last_decisions[30:]
            self.last_error = None
            self.cycles += 1
        self.write_status()

    def poll_once(self, *, emit: bool = True) -> list[CycleResult]:
        if self.universe and not self.recorded_closes:
            self.sync_universe()
            if emit:
                self.write_status()
        remaining: int | None = None
        if self.max_cycles is not None:
            remaining = max(0, self.max_cycles - self.cycles)
            if remaining == 0:
                return []
        results: list[CycleResult] = []

        def take(result: CycleResult | None) -> None:
            if result is None:
                return
            results.append(result)
            if emit:
                self._record_result(result)

        if self.recorded_closes:
            event = self.recorded_closes.pop(0)
            nested = event.get("data") if isinstance(event.get("data"), dict) else None
            symbol = str(
                event.get("s")
                or (nested or {}).get("s")
                or (nested or {}).get("k", {}).get("s")
                or event.get("k", {}).get("s")
                or self.symbols[0]
            ).upper()
            if symbol not in self.loops:
                symbol = self.symbols[0]
            take(self.loops[symbol].handle_ws_event(event))
            return results
        if self.recorded_klines is not None:
            for symbol in self._symbols_for_pass():
                if remaining is not None and len(results) >= remaining:
                    break
                self._ensure_loop(symbol)
                payload = _kline_payload_for_symbol(self.recorded_klines, symbol)
                take(self.loops[symbol].handle_rest_klines(payload, now_ms=self.now_ms))
            return results
        for symbol in self._symbols_for_pass():
            if remaining is not None and len(results) >= remaining:
                break
            # Before every symbol (at most every ~10s): a maker entry that
            # filled meanwhile gets its stop, and guards see real entries.
            self.reconcile()
            take(self.poll_live_symbol(symbol))
        return results

    def emit_universe_line(self) -> None:
        if not self.print_universe:
            return
        try:
            rows = fetch_universe_summary()
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {"ok": False, "type": "universe", "error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return
        payload = universe_payload(rows)
        wanted: dict[str, Any] = {s: None for s in self.symbols}
        for row in rows:
            if row.symbol in wanted:
                wanted[row.symbol] = {
                    "last": row.last,
                    "price_change_percent": row.price_change_percent,
                }
        compact = {
            "ok": True,
            "type": "universe",
            "authenticated": False,
            "count": payload["count"],
            "host": payload["host"],
            "path": payload["path"],
            "watch": wanted,
        }
        print(json.dumps(compact, ensure_ascii=False), flush=True)

    def _compact_decision(self, result: CycleResult) -> dict[str, Any]:
        payload = decision_payload(result)
        judgment = payload.get("judgment") or {}
        intent = payload.get("intent") or {}
        return {
            "ts": utc_now(),
            "symbol": payload.get("symbol"),
            "action": payload.get("action"),
            "jev_action": judgment.get("action"),
            "judgment_action": judgment.get("action"),
            "skip_reason": payload.get("skip_reason"),
            "should_trade_now": judgment.get("should_trade_now"),
            "qty": intent.get("qty"),
            "price": intent.get("limit_price"),
            "venue": None if payload.get("execution") is None else payload["execution"].get("venue"),
            "judge_ms": payload.get("judge_ms"),
            "model_skipped": bool(payload.get("model_skipped")),
        }

    def write_status(self, *, last_error: str | None = None) -> None:
        if last_error is not None:
            with self._state_lock:
                self.last_error = last_error
        if self.status_path is None:
            return
        entry_guards = None
        try:
            cfg = load_entry_guard_config()
            now = datetime.now(timezone.utc)
            entries_last_hour = 0
            loss_streak = 0
            last_loss_ts = None
            if self.ledger is not None:
                entries_last_hour = self.ledger.count_entries_since(now - timedelta(hours=1))
                loss_streak, last_loss_ts = loss_streak_from_closes(
                    self.ledger.recent_close_pnls(limit=64)
                )
            entry_guards = build_entry_guard_state(
                config=cfg,
                now=now,
                entries_last_hour=entries_last_hour,
                loss_streak=loss_streak,
                last_loss_ts=last_loss_ts,
            ).as_dict()
        except Exception as exc:  # noqa: BLE001 — status must not die on guard math
            log_event("entry_guard_status_error", error=f"{type(exc).__name__}: {exc}")
            entry_guards = None
        with self._state_lock:
            cycles = self.cycles
            symbols = list(self.symbols)
            decisions = list(self.last_decisions[:30])
            last_error = self.last_error
        wallet, day_start, wallet_error = wallet_view(self.wallet_box)
        payload = {
            "ok": True,
            "running": True,
            "pid": os.getpid(),
            "ts": utc_now(),
            "venue": self.venue,
            "follow_jev": self.follow_jev,
            "cycles": cycles,
            "watch": symbols,
            "universe": self.universe,
            "seconds_to_next_5m": seconds_to_next_5m(),
            "last_error": last_error or wallet_error or self.wallet_box.get("sync_error"),
            "last_decisions": decisions,
            "wallet": wallet,
            "day_start_equity_usdt": day_start,
            "entry_guards": entry_guards,
            "reconcile": self.last_reconcile,
            "decision_backend": self.decision_backend,
            "laya_checkpoint": self.laya_checkpoint if self.decision_backend == "laya" else None,
            "hint": "Jev на закрытии 5m. Вход только BUY/лонг; выход MARKET; стоп на бирже.",
        }
        try:
            write_json(self.status_path, payload)
        except OSError:
            pass

    def emit_heartbeat(self) -> None:
        self.write_status()
        if not self.heartbeat:
            return
        print(
            json.dumps(
                {
                    "type": "waiting",
                    "cycles": self.cycles,
                    "seconds_to_next_5m": seconds_to_next_5m(),
                    "symbols": {
                        symbol: {
                            "last_fired_open_ts": loop.last_fired_open_ts,
                            "invocations": loop.invocations,
                        }
                        for symbol, loop in self.loops.items()
                    },
                    "watch": self.symbols,
                    "universe": self.universe,
                    "hint": "Jev runs on the next 5m close for watch symbols. --fire-latest judges the last bar now.",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    def run(self) -> int:
        started = time.monotonic()
        last_summary = 0.0
        backoff = 1.0
        self.write_status()
        while True:
            if self.max_runtime is not None and time.monotonic() - started >= self.max_runtime:
                return 0
            if self.max_cycles is not None and self.cycles >= self.max_cycles:
                return 0
            try:
                results = self.poll_once()
                backoff = 1.0
            except Exception as exc:  # noqa: BLE001 — 24/7 reconnects; bounded runs fail fast
                err = f"{type(exc).__name__}: {exc}"
                self.write_status(last_error=err)
                if self.max_cycles is not None or self.max_runtime is not None or self.offline:
                    print(
                        json.dumps(
                            {"ok": False, "error": err},
                            ensure_ascii=False,
                        )
                    )
                    return 1
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            if self.max_cycles is not None and self.cycles >= self.max_cycles:
                return 0
            if not results:
                self.emit_heartbeat()
            if self.offline and not self.recorded_closes:
                return 0 if self.cycles else 1
            now = time.monotonic()
            if self.print_universe and now - last_summary >= self.summary_interval:
                self.emit_universe_line()
                last_summary = now
            if self.max_runtime is not None and now - started >= self.max_runtime:
                return 0
            sleep_for = 0.0 if self.offline else self.kline_poll
            if self.max_runtime is not None:
                remaining = self.max_runtime - (time.monotonic() - started)
                if remaining <= 0:
                    return 0
                if sleep_for:
                    sleep_for = min(sleep_for, remaining)
            if sleep_for:
                time.sleep(sleep_for)


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def load_answers(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("answers JSON must be an object")
    return data
