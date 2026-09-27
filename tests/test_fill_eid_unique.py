"""Unique (symbol, exchange_order_id) on fills; old duplicates never break startup."""
from __future__ import annotations

import sqlite3

from jev_trader.ledger import Ledger

SYM = "BTCUSDT"


def _index(path) -> bool:
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'uq_fills_sym_eid'").fetchone() is not None


def _fill(ledger: Ledger, cid: str, eid, **kw) -> bool:
    args = dict(ts="2026-09-27T00:00:00+00:00", client_order_id=cid, exchange_order_id=eid, symbol=SYM, action="buy_long", qty=1.0, price=100.0, venue="binance_testnet", commission_usdt=0.01)
    args.update(kw)
    return ledger.record_exchange_fill(**args)


def test_fresh_ledger_gets_the_index(tmp_path) -> None:
    Ledger(tmp_path / "l.sqlite")
    assert _index(tmp_path / "l.sqlite")


def test_old_duplicates_skip_the_index_and_start_anyway(tmp_path, capsys) -> None:
    path = tmp_path / "l.sqlite"
    ledger = Ledger(path)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP INDEX uq_fills_sym_eid")
    assert _fill(ledger, "x1", 55)
    with sqlite3.connect(path) as conn:  # duplicate written by an old version
        conn.execute("INSERT INTO fills (ts, client_order_id, symbol, action, position_side, qty, price, venue, exchange_order_id) SELECT ts, 'x2', symbol, action, position_side, qty, price, venue, exchange_order_id FROM fills")
    Ledger(path)  # must not raise
    assert not _index(path)
    assert "ledger_fill_eid_duplicates" in capsys.readouterr().err


def test_duplicate_insert_is_refused_not_raised(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    assert _fill(ledger, "x1", 55)
    ledger._has_fill = lambda conn, cid, eid: False  # type: ignore[method-assign]  # simulate a race past the check
    assert _fill(ledger, "x2", 55, action="close", net_pnl_usdt=5.0) is False
    with sqlite3.connect(tmp_path / "l.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 1
    # Same orderId on another symbol is a different order.
    assert _fill(ledger, "x3", 55, symbol="ETHUSDT")


def test_enrichment_onto_a_taken_order_id_counts_a_try(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.sqlite")
    assert _fill(ledger, "x1", 55)
    assert _fill(ledger, "x2", None, commission_usdt=None)
    with sqlite3.connect(tmp_path / "l.sqlite") as conn:
        fid = conn.execute("SELECT id FROM fills WHERE client_order_id = 'x2'").fetchone()[0]
    ledger.apply_exchange_enrichment(fid, exchange_order_id=55, price=100.0, qty=1.0, commission_usdt=0.02)  # no raise
    with sqlite3.connect(tmp_path / "l.sqlite") as conn:
        eid, tries = conn.execute("SELECT exchange_order_id, enrich_tries FROM fills WHERE id = ?", (fid,)).fetchone()
    assert eid is None and tries == 1
