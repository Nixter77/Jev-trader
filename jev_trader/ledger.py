from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_trader.models import CycleResult, ExecutionResult, JevJudgment, Position, TradeIntent


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
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
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
                    updated_ts TEXT NOT NULL
                )
                """
            )
            conn.commit()

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
                        entry_type, venue, status, detail_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
                )
            symbol = (result.state or {}).get("symbol") or (intent.symbol if intent else "")
            mark = _mark_price(result)
            if symbol:
                self._touch_position(conn, symbol, mark)
            if (
                intent is not None
                and execution is not None
                and execution.status in {"paper_recorded", "accepted"}
                and intent.qty > 0
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
    ) -> None:
        conn.execute(
            """
            INSERT INTO positions (
                symbol, side, size, entry, cash_usdt, realized_pnl_usdt, last_mark, updated_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                side = excluded.side,
                size = excluded.size,
                entry = excluded.entry,
                cash_usdt = excluded.cash_usdt,
                realized_pnl_usdt = excluded.realized_pnl_usdt,
                last_mark = excluded.last_mark,
                updated_ts = excluded.updated_ts
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
        price = _fill_price(result, mark)
        if price is None:
            return
        row = self._row_position(conn, symbol)
        side = str(row["side"]) if row else "FLAT"
        size = float(row["size"]) if row else 0.0
        entry = float(row["entry"]) if row and row["entry"] is not None else None
        cash = float(row["cash_usdt"]) if row else 10_000.0
        realized_total = float(row["realized_pnl_usdt"]) if row else 0.0
        pnl: float | None = None
        pos_side = side
        if result.action == "buy_long":
            pos_side = "LONG"
            size = float(intent.qty)
            entry = price
            side = "LONG"
        elif result.action == "sell_short":
            pos_side = "SHORT"
            size = float(intent.qty)
            entry = price
            side = "SHORT"
        elif result.action == "close":
            pos_side = side if side != "FLAT" else "LONG"
            qty = float(intent.qty)
            if entry is not None and side != "FLAT":
                pnl = _realized_pnl(side, entry, price, qty)
                cash = cash + pnl
                realized_total = realized_total + pnl
            side = "FLAT"
            size = 0.0
            entry = None
        else:
            return
        conn.execute(
            """
            INSERT INTO fills (
                ts, client_order_id, symbol, action, position_side, qty, price,
                venue, realized_pnl_usdt, cash_usdt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                intent.client_order_id,
                symbol,
                result.action,
                pos_side,
                float(intent.qty),
                price,
                execution.venue,
                pnl,
                cash,
            ),
        )
        self._upsert_position(
            conn,
            symbol=symbol,
            side=side,
            size=size,
            entry=entry,
            cash_usdt=cash,
            realized_pnl_usdt=realized_total,
            last_mark=price,
        )

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
        return Position(
            side=side,  # type: ignore[arg-type]
            size=float(row["size"] or 0.0),
            cash_usdt=float(row["cash_usdt"] if row["cash_usdt"] is not None else default_cash),
            entry=None if entry is None else float(entry),
            upnl_pct=upnl_pct,
        )

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
        "model": judgment.model,
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
