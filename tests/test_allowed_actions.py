from jev_trader.jev import allowed_actions_for_position, build_questions, position_side_from_compact
from jev_trader.laya_client import build_laya_questions
from jev_trader.models import CompactState

def test_flat_actions_exclude_close():
    assert allowed_actions_for_position("FLAT") == ("buy_long", "hold")
    qs = build_questions(allowed_actions=allowed_actions_for_position("FLAT"))
    assert set(qs["action"].criteria) == {"buy_long", "hold"}
    lq = build_laya_questions(allowed_actions=allowed_actions_for_position("FLAT"))
    assert set(lq["action"]["criteria"]) == {"buy_long", "hold"}

def test_long_actions_are_hold_or_close():
    assert allowed_actions_for_position("LONG") == ("hold", "close")
    qs = build_questions(allowed_actions=allowed_actions_for_position("LONG"))
    assert set(qs["action"].criteria) == {"hold", "close"}

def test_position_side_from_compact():
    c = CompactState(payload={"position": {"side": "LONG", "size": 1}}, text="pos=LONG")
    assert position_side_from_compact(c) == "LONG"
    c2 = CompactState(payload={}, text="x")
    assert position_side_from_compact(c2) == "FLAT"
