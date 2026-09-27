from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_trader.execution import fill_price, fill_qty, is_real_fill, wallet_positions_known
from jev_trader.models import CycleResult, ExecutionResult, JevJudgment, Position, TradeIntent
from jev_trader.status import log_event


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path: str | Path) -> None:
        raw = Path(path)
        # zsh line-wrap often turns `--ledger data/ledger.sqlite` into `--ledger data/`
        if str(path).endswith(("/", "\\")) or raw.is_dir():
            raw = raw / "ledger.sqlite"
        self.path = raw
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        # The blotter thread reads while the bot thread writes. WAL lets the
        # reader proceed; busy_timeout waits out the single writer instead of
        # raising "database is locked".
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    skip_reason TEXT,
                    risk_event TEXT,
                    judgment_json TEXT,
                    intent_json TEXT,
                    execution_json TEXT,
                    state_text TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    client_order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    qty REAL NOT NULL,
                    reduce_only INTEGER NOT NULL,
                    entry_type TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    client_order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    position_side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    venue TEXT NOT NULL,
                    realized_pnl_usdt REAL,
                    cash_usdt REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry REAL,
                    cash_usdt REAL NOT NULL,
                    realized_pnl_usdt REAL NOT NULL DEFAULT 0,
                    last_mark REAL,
                    updated_ts TEXT NOT NULL,
                    stop_price REAL
                )
                """
            )
            self._ensure_column(conn, "positions", "stop_price", "REAL")
            # Exchange reconciliation (orders resolved later, fills from userTrades).
            for column, decl in (
                ("stop_price", "REAL"),
                ("exchange_order_id", "TEXT"),
                ("final_status", "TEXT"),
                ("final_ts", "TEXT"),
            ):
                self._ensure_column(conn, "orders", column, decl)
            for column, decl in (
                ("exchange_order_id", "TEXT"),
                ("commission_usdt", "REAL"),
                ("funding_usdt", "REAL"),
                ("gross_pnl_usdt", "REAL"),
                ("source", "TEXT"),
                ("pnl_source", "TEXT"),
                ("enrich_tries", "INTEGER NOT NULL DEFAULT 0"),
                # Commission paid in a non-USDT asset (e.g. BNB): the USDT
                # commission / net PnL on this row is not exact.
                ("fee_unaccounted", "INTEGER NOT NULL DEFAULT 0"),
            ):
                self._ensure_column(conn, "fills", column, decl)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_cid ON fills(client_order_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_sym_ts ON fills(symbol, ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_cid ON orders(client_order_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts)")
            conn.commit()
        self._ensure_unique_fill_eid()

    def _ensure_unique_fill_eid(self) -> None:
        """One fill per exchange order (orderIds are per symbol on Binance).

        Old duplicates block the index: then only log, never fail startup.
        """
        try:
            with self._connect() as conn:
                have = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_fills_sym_eid'"
                ).fetchone()
                if have is not None:
                    return
                dups = conn.execute(
                    """
                    SELECT symbol, exchange_order_id, COUNT(*) AS n FROM fills
                    WHERE exchange_order_id IS NOT NULL
                    GROUP BY symbol, exchange_order_id HAVING COUNT(*) > 1
                    LIMIT 20
                    """
                ).fetchall()
                if dups:
                    log_event(
                        "ledger_fill_eid_duplicates",
                        path=str(self.path),
                        duplicates=[dict(r) for r in dups],
                        index="not_created",
                    )
                    return
                conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_fills_sym_eid
                    ON fills(symbol, exchange_order_id) WHERE exchange_order_id IS NOT NULL
                    """
                )
                conn.commit()
        except sqlite3.Error as exc:
            log_event("ledger_fill_eid_index_error", path=str(self.path), error=f"{type(exc).__name__}: {exc}")

    def _ensure_column(
        self, conn: sqlite3.Connection, table: str, column: str, decl: str
    ) -> None:
        cols = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def record(self, result: CycleResult) -> None:
        intent = result.intent
        execution = result.execution
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions (
                    ts, symbol, action, skip_reason, risk_event,
                    judgment_json, intent_json, execution_json, state_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    (result.state or {}).get("symbol") or (intent.symbol if intent else ""),
                    result.action,
                    result.skip_reason,
                    result.risk_event,
                    json.dumps(_judgment_dict(result.judgment), ensure_ascii=False),
                    json.dumps(_intent_dict(intent), ensure_ascii=False),
                    json.dumps(_exec_dict(execution), ensure_ascii=False),
                    result.state_text,
                ),
            )
            if intent is not None and execution is not None and execution.status != "skipped":
                conn.execute(
                    """
                    INSERT INTO orders (
                        ts, client_order_id, symbol, action, qty, reduce_only,
                        entry_type, venue, status, detail_json, stop_price,
                        exchange_order_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _now(),
                        intent.client_order_id,
                        intent.symbol,
                        intent.action,
                        intent.qty,
                        1 if intent.reduce_only else 0,
                        intent.entry_type,
                        execution.venue,
                        execution.status,
                        json.dumps(execution.detail, ensure_ascii=False),
                        intent.stop_price,
                        _exchange_order_id(execution),
                    ),
                )
            symbol = (result.state or {}).get("symbol") or (intent.symbol if intent else "")
            mark = _mark_price(result)
            if symbol:
                self._touch_position(conn, symbol, mark)
            if (
                intent is not None
                and execution is not None
                and is_real_fill(execution)
                and fill_qty(execution, intent.qty) > 0
                and result.action in {"buy_long", "sell_short", "close"}
            ):
                self._apply_fill(conn, result, mark)
            conn.commit()

    def _row_position(self, conn: sqlite3.Connection, symbol: str) -> sqlite3.Row | None:
        return conn.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,)).fetchone()

    def _upsert_position(
        self,
        conn: sqlite3.Connection,
        *,
        symbol: str,
        side: str,
        size: float,
        entry: float | None,
        cash_usdt: float,
        realized_pnl_usdt: float,
        last_mark: float | None,
        stop_price: float | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO positions (
                symbol, side, size, entry, cash_usdt, realized_pnl_usdt, last_mark, updated_ts,
                stop_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                side = excluded.side,
                size = excluded.size,
                entry = excluded.entry,
                cash_usdt = excluded.cash_usdt,
                realized_pnl_usdt = excluded.realized_pnl_usdt,
                last_mark = excluded.last_mark,
                updated_ts = excluded.updated_ts,
                stop_price = excluded.stop_price
            """,
            (
                symbol,
                side,
                size,
                entry,
                cash_usdt,
                realized_pnl_usdt,
                last_mark,
                _now(),
                stop_price,
            ),
        )

    def _touch_position(self, conn: sqlite3.Connection, symbol: str, mark: float | None) -> None:
        row = self._row_position(conn, symbol)
        if row is None:
            self._upsert_position(
                conn,
                symbol=symbol,
                side="FLAT",
                size=0.0,
                entry=None,
                cash_usdt=10_000.0,
                realized_pnl_usdt=0.0,
                last_mark=mark,
            )
            return
        if mark is None:
            return
        conn.execute(
            "UPDATE positions SET last_mark = ?, updated_ts = ? WHERE symbol = ?",
            (mark, _now(), symbol),
        )

    def _apply_fill(
        self,
        conn: sqlite3.Connection,
        result: CycleResult,
        mark: float | None,
    ) -> None:
        intent = result.intent
        execution = result.execution
        if intent is None or execution is None:
            return
        symbol = intent.symbol
        qty = fill_qty(execution, float(intent.qty))
        price = fill_price(execution, _fill_price(result, mark))
        if price is None or qty <= 0:
            return
        eid = _exchange_order_id(execution)
        if eid is not None and self._has_fill(conn, None, eid):
            # Already settled under this exchange orderId (e.g. the reconciler
            # recorded it as x<orderId> before a flatten in another process
            # wrote its own row). A second row would count the PnL twice.
            return
        row = self._row_position(conn, symbol)
        side = str(row["side"]) if row else "FLAT"
        size = float(row["size"]) if row else 0.0
        entry = float(row["entry"]) if row and row["entry"] is not None else None
        cash = float(row["cash_usdt"]) if row else 10_000.0
        realized_total = float(row["realized_pnl_usdt"]) if row else 0.0
        stop_price = None
        if row is not None:
            try:
                raw_stop = row["stop_price"]
            except (KeyError, IndexError):
                raw_stop = None
            stop_price = None if raw_stop is None else float(raw_stop)
        pnl: float | None = None
        pos_side = side
        if result.action == "buy_long":
            pos_side = "LONG"
            size = qty
            entry = price
            side = "LONG"
            stop_price = intent.stop_price
        elif result.action == "sell_short":
            pos_side = "SHORT"
            size = qty
            entry = price
            side = "SHORT"
            stop_price = None
        elif result.action == "close":
            pos_side = side if side != "FLAT" else "LONG"
            if entry is not None and side != "FLAT":
                pnl = _realized_pnl(side, entry, price, qty)
                cash = cash + pnl
                realized_total = realized_total + pnl
            side = "FLAT"
            size = 0.0
            entry = None
            stop_price = None
        else:
            return
        try:
            self._insert_bot_fill(conn, (
                _now(),
                intent.client_order_id,
                symbol,
                result.action,
                pos_side,
                qty,
                price,
                execution.venue,
                pnl,
                cash,
                _exchange_order_id(execution),
                None if pnl is None else "local",
            ))
        except sqlite3.IntegrityError as exc:
            # Same exchange order already settled: a second row would count
            # its PnL twice. The statement is aborted, the transaction goes on.
            log_event("ledger_fill_duplicate", symbol=symbol, eid=eid, error=str(exc))
            return
        self._upsert_position(
            conn,
            symbol=symbol,
            side=side,
            size=size,
            entry=entry,
            cash_usdt=cash,
            realized_pnl_usdt=realized_total,
            last_mark=price,
            stop_price=stop_price,
        )

    def set_stop(self, symbol: str, stop_price: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE positions
                SET stop_price = ?, updated_ts = ?
                WHERE symbol = ? AND side = 'LONG' AND size > 0
                """,
                (float(stop_price), _now(), symbol.upper()),
            )
            conn.commit()

    def seconds_since_last_close(self, symbol: str) -> float | None:
        """Seconds since the latest close fill for symbol, or None if never closed."""
        return self._seconds_since_fill(symbol, action="close")

    def seconds_since_last_entry(self, symbol: str) -> float | None:
        """Seconds since the latest buy_long fill for symbol, or None if never entered."""
        return self._seconds_since_fill(symbol, action="buy_long")

    def _seconds_since_fill(self, symbol: str, *, action: str) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT ts FROM fills
                WHERE symbol = ? AND action = ?
                ORDER BY ts DESC, id DESC LIMIT 1
                """,
                (symbol.upper(), action),
            ).fetchone()
        if row is None:
            return None
        raw = str(row["ts"])
        try:
            ts = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())

    def count_entries_since(self, since: datetime) -> int:
        """Entries (any symbol) at or after `since` (UTC).

        Real buy_long fills (bot or exchange-reconciled, by trade time) plus
        entry orders placed since `since` that are still unresolved (resting
        GTX / submit_unknown). A resting maker order can fill at any moment,
        so it counts against the hourly cap until the reconciler settles it.
        """
        bound = _iso_utc(since)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM fills
                    WHERE action = 'buy_long' AND ts >= ?)
                  +
                  (SELECT COUNT(*) FROM orders o
                    WHERE o.action = 'buy_long' AND o.ts >= ?
                      AND o.status IN ('working', 'accepted', 'submit_unknown')
                      AND o.final_status IS NULL
                      AND NOT EXISTS (
                        SELECT 1 FROM fills f WHERE f.client_order_id = o.client_order_id
                      )) AS n
                """,
                (bound, bound),
            ).fetchone()
        return int(row["n"] if row is not None else 0)

    # --- exchange reconciliation -------------------------------------------

    def pending_orders(self, since: datetime) -> list[dict[str, Any]]:
        """Orders whose outcome the exchange still has to tell us, oldest first.

        buy_long that were resting (working / accepted) or whose submit timed
        out, and closes whose submit timed out. Rows already settled or with a
        fill under the same clientOrderId are excluded.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT o.* FROM orders o
                WHERE o.ts >= ?
                  AND o.final_status IS NULL
                  AND (
                    (o.action = 'buy_long'
                      AND o.status IN ('working', 'accepted', 'submit_unknown'))
                    OR (o.action = 'close' AND o.status = 'submit_unknown')
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM fills f WHERE f.client_order_id = o.client_order_id
                  )
                ORDER BY o.ts, o.id
                """,
                (_iso_utc(since),),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_order_final(
        self, order_row_id: int, status: str, *, exchange_order_id: Any = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE orders
                SET final_status = ?, final_ts = ?,
                    exchange_order_id = COALESCE(?, exchange_order_id)
                WHERE id = ?
                """,
                (
                    str(status),
                    _now(),
                    None if exchange_order_id is None else str(exchange_order_id),
                    int(order_row_id),
                ),
            )
            conn.commit()

    def order_stop_price(self, client_order_id: str) -> float | None:
        """Stop the entry was sized with: orders.stop_price, else the decision intent."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT stop_price FROM orders
                WHERE client_order_id = ? AND stop_price IS NOT NULL
                ORDER BY id DESC LIMIT 1
                """,
                (client_order_id,),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    """
                    SELECT json_extract(intent_json, '$.stop_price') AS stop_price
                    FROM decisions
                    WHERE json_extract(intent_json, '$.client_order_id') = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (client_order_id,),
                ).fetchone()
        if row is None or row["stop_price"] is None:
            return None
        try:
            value = float(row["stop_price"])
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def has_fill(self, *, client_order_id: str | None = None, exchange_order_id: Any = None) -> bool:
        with self._connect() as conn:
            return self._has_fill(conn, client_order_id, exchange_order_id)

    def _has_fill(
        self, conn: sqlite3.Connection, client_order_id: str | None, exchange_order_id: Any
    ) -> bool:
        if client_order_id:
            if conn.execute(
                "SELECT 1 FROM fills WHERE client_order_id = ? LIMIT 1", (client_order_id,)
            ).fetchone():
                return True
        if exchange_order_id is not None:
            if conn.execute(
                "SELECT 1 FROM fills WHERE exchange_order_id = ? LIMIT 1",
                (str(exchange_order_id),),
            ).fetchone():
                return True
        return False

    @staticmethod
    def _insert_bot_fill(conn: sqlite3.Connection, values: tuple[Any, ...]) -> None:
        conn.execute(
            """
            INSERT INTO fills (
                ts, client_order_id, symbol, action, position_side, qty, price,
                venue, realized_pnl_usdt, cash_usdt, exchange_order_id, source,
                pnl_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'bot', ?)
            """,
            values,
        )

    def record_exchange_fill(
        self,
        *,
        ts: str,
        client_order_id: str,
        exchange_order_id: Any,
        symbol: str,
        action: str,
        qty: float,
        price: float,
        venue: str,
        commission_usdt: float | None,
        gross_pnl_usdt: float | None = None,
        funding_usdt: float | None = None,
        net_pnl_usdt: float | None = None,
        fee_unaccounted: bool = False,
    ) -> bool:
        """Insert a fill the bot did not see (maker entry filled later, exchange
        stop / manual close). Deduped by clientOrderId and exchange orderId.

        Positions side/size come from the wallet sync, not from here; a close
        only adds its PnL to the running totals. Returns False on a duplicate.
        """
        symbol = symbol.upper()
        try:
            return self._record_exchange_fill(
                ts=ts, client_order_id=client_order_id, exchange_order_id=exchange_order_id,
                symbol=symbol, action=action, qty=qty, price=price, venue=venue,
                commission_usdt=commission_usdt, gross_pnl_usdt=gross_pnl_usdt,
                funding_usdt=funding_usdt, net_pnl_usdt=net_pnl_usdt, fee_unaccounted=fee_unaccounted,
            )
        except sqlite3.IntegrityError as exc:
            # The connection context rolled back the PnL update with the insert.
            log_event("ledger_fill_duplicate", symbol=symbol, eid=exchange_order_id, cid=client_order_id, error=str(exc))
            return False

    def _record_exchange_fill(
        self,
        *,
        ts: str,
        client_order_id: str,
        exchange_order_id: Any,
        symbol: str,
        action: str,
        qty: float,
        price: float,
        venue: str,
        commission_usdt: float | None,
        gross_pnl_usdt: float | None,
        funding_usdt: float | None,
        net_pnl_usdt: float | None,
        fee_unaccounted: bool,
    ) -> bool:
        with self._connect() as conn:
            if self._has_fill(conn, client_order_id, exchange_order_id):
                return False
            row = self._row_position(conn, symbol)
            cash = float(row["cash_usdt"]) if row else 10_000.0
            realized_total = float(row["realized_pnl_usdt"]) if row else 0.0
            if action == "close" and net_pnl_usdt is not None:
                cash += float(net_pnl_usdt)
                realized_total += float(net_pnl_usdt)
                if row is not None:
                    conn.execute(
                        """
                        UPDATE positions SET cash_usdt = ?, realized_pnl_usdt = ?, updated_ts = ?
                        WHERE symbol = ?
                        """,
                        (cash, realized_total, _now(), symbol),
                    )
            conn.execute(
                """
                INSERT INTO fills (
                    ts, client_order_id, symbol, action, position_side, qty, price,
                    venue, realized_pnl_usdt, cash_usdt, exchange_order_id,
                    commission_usdt, funding_usdt, gross_pnl_usdt, source, pnl_source,
                    fee_unaccounted
                ) VALUES (?, ?, ?, ?, 'LONG', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'exchange', ?, ?)
                """,
                (
                    ts,
                    client_order_id,
                    symbol,
                    action,
                    float(qty),
                    float(price),
                    venue,
                    net_pnl_usdt if action == "close" else None,
                    cash,
                    None if exchange_order_id is None else str(exchange_order_id),
                    commission_usdt,
                    funding_usdt,
                    gross_pnl_usdt,
                    _exchange_pnl_source(fee_unaccounted) if action == "close" and net_pnl_usdt is not None else None,
                    1 if fee_unaccounted else 0,
                ),
            )
            conn.commit()
        return True

    def fills_to_enrich(self, since: datetime, *, limit: int = 10, max_tries: int = 5) -> list[dict[str, Any]]:
        """Bot fills without exchange commission yet, oldest first, with the orderId."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.*,
                  COALESCE(
                    f.exchange_order_id,
                    (SELECT COALESCE(o.exchange_order_id,
                                     CAST(json_extract(o.detail_json, '$.body.orderId') AS TEXT))
                       FROM orders o WHERE o.client_order_id = f.client_order_id
                       ORDER BY o.id DESC LIMIT 1)
                  ) AS order_eid
                FROM fills f
                WHERE f.ts >= ?
                  AND f.venue != 'paper'
                  AND f.commission_usdt IS NULL
                  AND COALESCE(f.enrich_tries, 0) < ?
                ORDER BY f.ts, f.id
                LIMIT ?
                """,
                (_iso_utc(since), int(max_tries), int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def bump_enrich_tries(self, fill_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE fills SET enrich_tries = COALESCE(enrich_tries, 0) + 1 WHERE id = ?",
                (int(fill_id),),
            )
            conn.commit()

    def apply_exchange_enrichment(
        self,
        fill_id: int,
        *,
        exchange_order_id: Any,
        price: float,
        qty: float,
        commission_usdt: float,
        gross_pnl_usdt: float | None = None,
        funding_usdt: float | None = None,
        net_pnl_usdt: float | None = None,
        fee_unaccounted: bool = False,
    ) -> None:
        """Replace a bot fill's self-computed numbers with the exchange's.

        For a close the net PnL replaces realized_pnl_usdt and the running
        position totals move by the difference.
        """
        try:
            self._apply_exchange_enrichment(
                fill_id, exchange_order_id=exchange_order_id, price=price, qty=qty,
                commission_usdt=commission_usdt, gross_pnl_usdt=gross_pnl_usdt,
                funding_usdt=funding_usdt, net_pnl_usdt=net_pnl_usdt, fee_unaccounted=fee_unaccounted,
            )
        except sqlite3.IntegrityError as exc:
            # Another row already owns this exchange order: leave both alone,
            # count the try so it is not retried forever.
            log_event("ledger_enrich_duplicate", fill_id=int(fill_id), eid=exchange_order_id, error=str(exc))
            self.bump_enrich_tries(int(fill_id))

    def _apply_exchange_enrichment(
        self,
        fill_id: int,
        *,
        exchange_order_id: Any,
        price: float,
        qty: float,
        commission_usdt: float,
        gross_pnl_usdt: float | None,
        funding_usdt: float | None,
        net_pnl_usdt: float | None,
        fee_unaccounted: bool,
    ) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM fills WHERE id = ?", (int(fill_id),)).fetchone()
            if row is None:
                return
            conn.execute(
                """
                UPDATE fills SET exchange_order_id = ?, price = ?, qty = ?, commission_usdt = ?,
                    fee_unaccounted = ?, enrich_tries = COALESCE(enrich_tries, 0) + 1
                WHERE id = ?
                """,
                (
                    None if exchange_order_id is None else str(exchange_order_id),
                    float(price),
                    float(qty),
                    float(commission_usdt),
                    1 if fee_unaccounted else 0,
                    int(fill_id),
                ),
            )
            if row["action"] == "close" and net_pnl_usdt is not None:
                old = float(row["realized_pnl_usdt"] or 0.0)
                delta = float(net_pnl_usdt) - old
                conn.execute(
                    """
                    UPDATE fills SET realized_pnl_usdt = ?, gross_pnl_usdt = ?, funding_usdt = ?,
                        pnl_source = ?, cash_usdt = COALESCE(cash_usdt, 0) + ?
                    WHERE id = ?
                    """,
                    (
                        float(net_pnl_usdt),
                        gross_pnl_usdt,
                        funding_usdt,
                        _exchange_pnl_source(fee_unaccounted),
                        delta,
                        int(fill_id),
                    ),
                )
                conn.execute(
                    """
                    UPDATE positions SET realized_pnl_usdt = realized_pnl_usdt + ?,
                        cash_usdt = cash_usdt + ?, updated_ts = ?
                    WHERE symbol = ?
                    """,
                    (delta, delta, _now(), str(row["symbol"]).upper()),
                )
            conn.commit()

    def entry_fill_before(self, symbol: str, ts: str) -> dict[str, Any] | None:
        """Latest buy_long fill for symbol at or before ts."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM fills
                WHERE symbol = ? AND action = 'buy_long' AND ts <= ?
                ORDER BY ts DESC, id DESC LIMIT 1
                """,
                (symbol.upper(), ts),
            ).fetchone()
        return None if row is None else dict(row)

    def close_ts_before(self, symbol: str, ts: str) -> str | None:
        """ts of the latest close fill for symbol strictly before ts."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT ts FROM fills
                WHERE symbol = ? AND action = 'close' AND ts < ?
                ORDER BY ts DESC, id DESC LIMIT 1
                """,
                (symbol.upper(), ts),
            ).fetchone()
        return None if row is None else str(row["ts"])

    def open_entries(self, since: datetime) -> list[dict[str, Any]]:
        """Per symbol, the latest fill when it is a buy_long placed since `since`
        (an entry the ledger has no close for yet)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.* FROM fills f
                WHERE f.ts >= ?
                  AND f.action = 'buy_long'
                  AND NOT EXISTS (
                    SELECT 1 FROM fills g
                    WHERE g.symbol = f.symbol
                      AND (g.ts > f.ts OR (g.ts = f.ts AND g.id > f.id))
                  )
                ORDER BY f.ts
                """,
                (_iso_utc(since),),
            ).fetchall()
        return [dict(r) for r in rows]

    def known_order_ids(self, symbol: str, since_ts: str) -> set[str]:
        """Exchange orderIds already behind a fill for symbol since since_ts."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(
                    f.exchange_order_id,
                    (SELECT COALESCE(o.exchange_order_id,
                                     CAST(json_extract(o.detail_json, '$.body.orderId') AS TEXT))
                       FROM orders o WHERE o.client_order_id = f.client_order_id
                       ORDER BY o.id DESC LIMIT 1)
                ) AS eid
                FROM fills f
                WHERE f.symbol = ? AND f.ts >= ?
                """,
                (symbol.upper(), since_ts),
            ).fetchall()
        return {str(r["eid"]) for r in rows if r["eid"] is not None}

    def recent_close_pnls(self, *, limit: int = 64) -> list[tuple[datetime, float]]:
        """Newest-first close fills with realized PnL. Break-even is 0.0 when null."""
        lim = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, realized_pnl_usdt FROM fills
                WHERE action = 'close'
                ORDER BY ts DESC, id DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
        out: list[tuple[datetime, float]] = []
        for row in rows:
            raw = str(row["ts"])
            try:
                ts = datetime.fromisoformat(raw)
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            pnl_raw = row["realized_pnl_usdt"]
            pnl = 0.0 if pnl_raw is None else float(pnl_raw)
            out.append((ts, pnl))
        return out

    def count_open_positions(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM positions WHERE side != 'FLAT' AND size > 0"
            ).fetchone()
        return int(row["n"] if row is not None else 0)

    def open_symbols(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol FROM positions WHERE side != 'FLAT' AND size > 0 ORDER BY symbol"
            ).fetchall()
        return [str(row["symbol"]).upper() for row in rows]

    def load_position(self, symbol: str, default_cash: float = 10_000.0) -> Position:
        with self._connect() as conn:
            row = self._row_position(conn, symbol.upper())
        if row is None:
            return Position(side="FLAT", size=0.0, cash_usdt=default_cash)
        side = str(row["side"])
        if side not in {"FLAT", "LONG", "SHORT"}:
            side = "FLAT"
        mark = row["last_mark"]
        entry = row["entry"]
        upnl_pct = None
        if side != "FLAT" and entry and mark:
            upnl_pct = (float(mark) - float(entry)) / float(entry) * 100.0
            if side == "SHORT":
                upnl_pct = -upnl_pct
        stop_raw = None
        try:
            stop_raw = row["stop_price"]
        except (KeyError, IndexError):
            stop_raw = None
        return Position(
            side=side,  # type: ignore[arg-type]
            size=float(row["size"] or 0.0),
            cash_usdt=float(row["cash_usdt"] if row["cash_usdt"] is not None else default_cash),
            entry=None if entry is None else float(entry),
            upnl_pct=upnl_pct,
            stop_price=None if stop_raw is None else float(stop_raw),
        )

    def sync_exchange_positions(self, wallet: dict[str, Any]) -> None:
        """Overwrite local side/size/entry from Binance wallet. Do not invent fills.

        A wallet without a positions list (balance fallback) is skipped: its
        empty list would mark every open position FLAT.
        """
        if not wallet_positions_known(wallet):
            return
        cash = float(wallet.get("equity_usdt") or 10_000.0)
        seen: set[str] = set()
        with self._connect() as conn:
            for raw in wallet.get("positions") or []:
                if not isinstance(raw, dict):
                    continue
                symbol = str(raw.get("symbol") or "").upper()
                size = float(raw.get("size") or 0.0)
                side = str(raw.get("side") or "FLAT")
                if not symbol or size <= 0 or side not in {"LONG", "SHORT"}:
                    continue
                seen.add(symbol)
                row = self._row_position(conn, symbol)
                realized = float(row["realized_pnl_usdt"]) if row else 0.0
                stop_price = None
                if row is not None and side == "LONG":
                    try:
                        stop_raw = row["stop_price"]
                    except (KeyError, IndexError):
                        stop_raw = None
                    stop_price = None if stop_raw is None else float(stop_raw)
                mark = raw.get("entry")
                self._upsert_position(
                    conn,
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry=None if not raw.get("entry") else float(raw["entry"]),
                    cash_usdt=cash,
                    realized_pnl_usdt=realized,
                    last_mark=None if mark is None else float(mark),
                    stop_price=stop_price,
                )
            rows = conn.execute(
                "SELECT symbol, side, size FROM positions WHERE side != 'FLAT' AND size > 0"
            ).fetchall()
            for row in rows:
                symbol = str(row["symbol"]).upper()
                if symbol in seen:
                    continue
                existing = self._row_position(conn, symbol)
                realized = float(existing["realized_pnl_usdt"]) if existing else 0.0
                self._upsert_position(
                    conn,
                    symbol=symbol,
                    side="FLAT",
                    size=0.0,
                    entry=None,
                    cash_usdt=cash,
                    realized_pnl_usdt=realized,
                    last_mark=None if existing is None else existing["last_mark"],
                    stop_price=None,
                )
            conn.commit()

    def book(self, *, limit: int = 20) -> dict[str, Any]:
        with self._connect() as conn:
            positions = [dict(r) for r in conn.execute("SELECT * FROM positions ORDER BY symbol")]
            fills = [
                dict(r)
                for r in conn.execute("SELECT * FROM fills ORDER BY id DESC LIMIT ?", (limit,))
            ]
            decisions = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT id, ts, symbol, action, skip_reason, risk_event,
                           judgment_json, intent_json, execution_json
                    FROM decisions ORDER BY id DESC LIMIT ?
                    """,
                    (limit,),
                )
            ]
        realized = sum(float(p.get("realized_pnl_usdt") or 0.0) for p in positions)
        unrealized = 0.0
        open_positions = []
        for pos in positions:
            mark = pos.get("last_mark")
            entry = pos.get("entry")
            size = float(pos.get("size") or 0.0)
            side = pos.get("side") or "FLAT"
            upnl = None
            if side != "FLAT" and mark is not None and entry is not None and size:
                upnl = _realized_pnl(str(side), float(entry), float(mark), size)
                unrealized += upnl
            open_positions.append(
                {
                    "symbol": pos.get("symbol"),
                    "side": side,
                    "size": size,
                    "entry": entry,
                    "last_mark": mark,
                    "unrealized_pnl_usdt": upnl,
                    "realized_pnl_usdt": pos.get("realized_pnl_usdt"),
                    "cash_usdt": pos.get("cash_usdt"),
                    "stop_price": pos.get("stop_price"),
                    "updated_ts": pos.get("updated_ts"),
                }
            )

        def _parse(raw: str | None) -> Any:
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw

        recent = []
        for row in decisions:
            judgment = _parse(row.get("judgment_json"))
            intent = _parse(row.get("intent_json"))
            execution = _parse(row.get("execution_json"))
            recent.append(
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "symbol": row["symbol"],
                    "action": row["action"],
                    "skip_reason": row["skip_reason"],
                    "judgment_action": None if not isinstance(judgment, dict) else judgment.get("action"),
                    "should_trade_now": None
                    if not isinstance(judgment, dict)
                    else judgment.get("should_trade_now"),
                    "qty": None if not isinstance(intent, dict) else intent.get("qty"),
                    "price": None if not isinstance(intent, dict) else intent.get("limit_price"),
                    "stop_price": None if not isinstance(intent, dict) else intent.get("stop_price"),
                    "venue": None if not isinstance(execution, dict) else execution.get("venue"),
                    "execution_status": None if not isinstance(execution, dict) else execution.get("status"),
                }
            )
        return {
            "ok": True,
            "ledger": str(self.path),
            "positions": open_positions,
            "realized_pnl_usdt": realized,
            "unrealized_pnl_usdt": unrealized,
            "fills": fills,
            "recent_decisions": recent,
        }


def _exchange_pnl_source(fee_unaccounted: bool) -> str:
    """'exchange' only when every commission was in USDT."""
    return "exchange_fee_unaccounted" if fee_unaccounted else "exchange"


def _iso_utc(when: datetime) -> str:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).isoformat()


def _exchange_order_id(execution: ExecutionResult | None) -> str | None:
    detail = getattr(execution, "detail", None)
    body = detail.get("body") if isinstance(detail, dict) else None
    if isinstance(body, dict) and body.get("orderId") is not None:
        return str(body["orderId"])
    return None


def _mark_price(result: CycleResult) -> float | None:
    price = ((result.state or {}).get("price") or {}).get("close")
    if price is not None:
        return float(price)
    intent = result.intent
    if intent is not None and intent.limit_price:
        return float(intent.limit_price)
    return None


def _fill_price(result: CycleResult, mark: float | None) -> float | None:
    intent = result.intent
    if intent is not None and intent.limit_price is not None and intent.limit_price > 0:
        return float(intent.limit_price)
    return mark


def _realized_pnl(side: str, entry: float, exit_px: float, qty: float) -> float:
    if side == "LONG":
        return (exit_px - entry) * qty
    if side == "SHORT":
        return (entry - exit_px) * qty
    return 0.0


def _judgment_dict(judgment: JevJudgment | None) -> dict[str, Any]:
    if judgment is None:
        return {}
    return {
        "action": judgment.action,
        "trend_aligned": judgment.trend_aligned,
        "false_break_risk": judgment.false_break_risk,
        "signal_strength": judgment.signal_strength,
        "should_trade_now": judgment.should_trade_now,
        "action_probabilities": dict(judgment.action_probabilities or {}),
        "signal_strength_score": judgment.signal_strength_score,
        "model": judgment.model,
        "raw": dict(judgment.raw or {}),
    }


def _intent_dict(intent: TradeIntent | None) -> dict[str, Any]:
    if intent is None:
        return {}
    return {
        "action": intent.action,
        "qty": intent.qty,
        "stop_price": intent.stop_price,
        "stop_distance": intent.stop_distance,
        "entry_type": intent.entry_type,
        "reduce_only": intent.reduce_only,
        "order_side": intent.order_side,
        "limit_price": intent.limit_price,
        "client_order_id": intent.client_order_id,
        "skip_reason": intent.skip_reason,
        "risk_event": intent.risk_event,
        "risk_pct": intent.risk_pct,
    }


def _exec_dict(execution: ExecutionResult | None) -> dict[str, Any]:
    if execution is None:
        return {}
    return {
        "status": execution.status,
        "venue": execution.venue,
        "client_order_id": execution.client_order_id,
        "reduce_only": execution.reduce_only,
        "detail": execution.detail,
    }
