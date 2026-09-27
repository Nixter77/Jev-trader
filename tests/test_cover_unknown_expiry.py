"""An unknown plain-BUY cover older than 72 h stops blocking covers: cancel by cid, settle rejected."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from jev_trader.ledger import Ledger
from tests.test_cover_own_id import REDUCE_ONLY_REJECTED, SYM, Exchange, _broker, _cover, _record

FILLED = (200, {"orderId": 6, "status": "FILLED", "executedQty": "1", "avgPrice": "99"})
OLD = 73 * 3600.0


def test_old_ledger_cover_is_expired_and_settled() -> None:
    ex = Exchange()
    ex.post_answers = [REDUCE_ONLY_REJECTED, FILLED]
    broker = _broker(ex)
    settled: list[tuple[str, str]] = []
    broker.unknown_cover_cids_fn = lambda symbol: [("jf_old", OLD)]
    broker.settle_unknown_cover_fn = lambda cid, status: settled.append((cid, status))
    result = broker.submit(_cover("jc_new"))
    assert result.status == "filled"
    assert settled == [("jf_old", "rejected")]
    assert any(m == "DELETE" and p.get("origClientOrderId") == "jf_old" for m, _path, p in ex.calls)


def test_young_ledger_cover_still_blocks() -> None:
    ex = Exchange()
    ex.post_answers = [REDUCE_ONLY_REJECTED]
    broker = _broker(ex)
    settled: list[tuple[str, str]] = []
    broker.unknown_cover_cids_fn = lambda symbol: [("jf_old", 3600.0)]
    broker.settle_unknown_cover_fn = lambda cid, status: settled.append((cid, status))
    result = broker.submit(_cover("jc_new"))
    assert result.detail["error"] == "cover_submit_unknown"
    assert settled == []


def test_plain_cid_list_keeps_the_old_behavior() -> None:
    ex = Exchange()
    ex.post_answers = [REDUCE_ONLY_REJECTED]
    broker = _broker(ex)
    broker.unknown_cover_cids_fn = lambda symbol: ["jf_old"]
    assert broker.submit(_cover()).detail["pending_client_order_id"] == "jf_old"


def test_ledger_end_to_end(tmp_path) -> None:
    ex = Exchange()
    ex.post_answers = [REDUCE_ONLY_REJECTED, (503, {"error": "down"})]
    broker = _broker(ex)
    ledger = Ledger(tmp_path / "l.sqlite")
    intent = _cover("jc_s1")
    first = broker.submit(intent)
    assert first.status == "submit_unknown"
    _record(ledger, intent, first)
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=73)).isoformat()
    with ledger._connect() as conn:
        conn.execute("UPDATE orders SET ts = ?", (old_ts,))
        conn.commit()
    [(cid, age)] = ledger.unknown_cover_rows(SYM)
    assert cid == first.client_order_id and age == pytest.approx(OLD, abs=60)

    fresh = _broker(ex)  # after a restart: only the ledger knows the cover
    fresh.unknown_cover_cids_fn = ledger.unknown_cover_rows
    fresh.settle_unknown_cover_fn = ledger.mark_order_final_by_cid
    ex.post_answers = [REDUCE_ONLY_REJECTED, FILLED]
    second = fresh.submit(_cover("jc_s2"))
    assert second.status == "filled"
    assert ledger.unknown_cover_cids(SYM) == []
    with ledger._connect() as conn:
        final = conn.execute("SELECT final_status FROM orders WHERE client_order_id = ?", (cid,)).fetchone()[0]
    assert final == "rejected"
