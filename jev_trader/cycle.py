from __future__ import annotations

import json
from typing import Any

from jev_trader.execution import (
    BinanceFuturesBroker,
    BinanceTestnetBroker,
    PaperBroker,
    fill_qty,
    is_real_fill,
)
from jev_trader.features import compute_features
from jev_trader.jev import JevClient, judgment_from_dict
from jev_trader.ledger import Ledger
from jev_trader.models import (
    AccountState,
    CycleResult,
    JevJudgment,
    MarketSnapshot,
    TradeIntent,
)
from jev_trader.policy import apply_policy
from jev_trader.risk import apply_risk
from jev_trader.state import build_compact_state
from jev_trader.telegram import TelegramNotifier


def intent_fields(intent: TradeIntent | None) -> dict[str, Any] | None:
    if intent is None or intent.skip_reason or intent.qty <= 0:
        return None
    return {
        "qty": intent.qty,
        "stop_price": intent.stop_price,
        "stop_distance": intent.stop_distance,
        "entry_type": intent.entry_type,
        "reduce_only": intent.reduce_only,
        "order_side": intent.order_side,
        "limit_price": intent.limit_price,
        "client_order_id": intent.client_order_id,
        "risk_pct": intent.risk_pct,
        "risk_event": intent.risk_event,
    }


def decision_payload(result: CycleResult) -> dict[str, Any]:
    intent = intent_fields(result.intent)
    symbol = (result.state or {}).get("symbol") or (result.intent.symbol if result.intent else None)
    payload: dict[str, Any] = {
        "symbol": symbol,
        "action": result.action,
        "skip_reason": result.skip_reason,
        "risk_event": result.risk_event,
        "intent": intent,
        "execution": None
        if result.execution is None
        else {
            "status": result.execution.status,
            "venue": result.execution.venue,
            "reduce_only": result.execution.reduce_only,
            "client_order_id": result.execution.client_order_id,
        },
        "judgment": None
        if result.judgment is None
        else {
            "action": result.judgment.action,
            "trend_aligned": result.judgment.trend_aligned,
            "false_break_risk": result.judgment.false_break_risk,
            "signal_strength": result.judgment.signal_strength,
            "should_trade_now": result.judgment.should_trade_now,
            "model": result.judgment.model,
        },
        "state_text": result.state_text,
    }
    return payload


def run_once(
    snapshot: MarketSnapshot,
    *,
    judgment: JevJudgment | None = None,
    jev_client: JevClient | None = None,
    account: AccountState | None = None,
    broker: PaperBroker | BinanceTestnetBroker | BinanceFuturesBroker | None = None,
    ledger: Ledger | None = None,
    notifier: TelegramNotifier | None = None,
    typesafe_api_key: str | None = None,
    follow_jev: bool = False,
    min_should_trade: float | None = None,
) -> CycleResult:
    """One paper/testnet decision cycle: features → state → Jev → policy → risk → exec."""
    features = compute_features(snapshot)
    compact = build_compact_state(snapshot, features)
    resolved = judgment
    if resolved is None:
        client = jev_client
        owns = False
        if client is None:
            if not typesafe_api_key:
                raise ValueError("Jev judgment missing and TYPESAFE_API_KEY not provided")
            client = JevClient(api_key=typesafe_api_key)
            owns = True
        try:
            resolved = client.judge(compact)
        finally:
            if owns:
                client.close()

    policy_kwargs: dict[str, Any] = {"follow_jev": follow_jev}
    if min_should_trade is not None:
        policy_kwargs["min_should_trade"] = min_should_trade
    policy = apply_policy(resolved, **policy_kwargs)
    acct = account or AccountState(
        equity_usdt=snapshot.position.cash_usdt,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=0 if snapshot.position.side == "FLAT" else 1,
    )
    intent = apply_risk(policy, features, snapshot, acct)
    exec_broker = broker or PaperBroker()
    execution = exec_broker.submit(intent)
    if (
        is_real_fill(execution)
        and intent.action == "buy_long"
        and intent.stop_price
        and fill_qty(execution, intent.qty) > 0
    ):
        place_stop = getattr(exec_broker, "place_stop_market", None)
        if callable(place_stop):
            try:
                stop_res = place_stop(
                    intent.symbol,
                    stop_price=float(intent.stop_price),
                    order_side="SELL",
                )
                if isinstance(execution.detail, dict):
                    execution.detail["stop"] = stop_res
            except Exception:  # noqa: BLE001 — fill already happened; stop is extra
                pass

    action = intent.action if intent.skip_reason is None else "hold"
    skip_reason = intent.skip_reason
    if policy.passed and intent.skip_reason is None:
        action = intent.action
        skip_reason = None
    elif not policy.passed:
        action = "hold"
        skip_reason = policy.skip_reason
    if execution.status == "unfilled":
        action = "hold"
        skip_reason = skip_reason or "unfilled"
    elif execution.status == "working":
        skip_reason = skip_reason or "working"

    result = CycleResult(
        action=action,
        skip_reason=skip_reason,
        intent=intent,
        execution=execution,
        judgment=resolved,
        state_text=compact.as_text(),
        state=compact.as_dict(),
        risk_event=intent.risk_event,
    )
    if ledger is not None:
        ledger.record(result)
    if notifier is not None and notifier.enabled:
        payload = decision_payload(result)
        notifier.send(
            f"jev {snapshot.symbol} action={payload['action']} "
            f"skip={payload['skip_reason']} venue={(execution.venue if execution else 'n/a')}"
        )
    return result


def run_once_from_answers(
    snapshot: MarketSnapshot,
    answers: dict[str, Any],
    **kwargs: Any,
) -> CycleResult:
    return run_once(snapshot, judgment=judgment_from_dict(answers), **kwargs)


def dumps_decision(result: CycleResult) -> str:
    return json.dumps(decision_payload(result), ensure_ascii=False, indent=2)
