from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from jev_trader.ledger import Ledger


def test_seconds_since_last_close_reads_latest_fill(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    # Use internal insert compatible with schema: record via SQL for isolation.
    now = datetime.now(timezone.utc)
    older = (now - timedelta(seconds=600)).isoformat()
    newer = (now - timedelta(seconds=120)).isoformat()
    with ledger._connect() as conn:
        conn.execute(
            """
            INSERT INTO fills (ts, client_order_id, symbol, action, position_side, qty, price, venue, realized_pnl_usdt, cash_usdt)
            VALUES (?, 'c1', 'BTCUSDT', 'close', 'LONG', 1.0, 100.0, 'test', 0.0, 10000.0)
            """,
            (older,),
        )
        conn.execute(
            """
            INSERT INTO fills (ts, client_order_id, symbol, action, position_side, qty, price, venue, realized_pnl_usdt, cash_usdt)
            VALUES (?, 'c2', 'BTCUSDT', 'close', 'LONG', 1.0, 101.0, 'test', 0.0, 10000.0)
            """,
            (newer,),
        )
        conn.commit()
    secs = ledger.seconds_since_last_close("btcusdt")
    assert secs is not None
    assert 100 <= secs <= 180
    assert ledger.seconds_since_last_close("ETHUSDT") is None
