from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from jev_trader.execution import (
    BinanceFuturesBroker,
    BinanceTestnetBroker,
    PaperBroker,
    client_order_id,
    current_flatten_generation,
    fill_qty,
    is_real_fill,
    order_lock,
)
from jev_trader.features import compute_features
from jev_trader.jev import JevClient, judgment_from_dict
from jev_trader.ledger import Ledger
from jev_trader.models import (
    AccountState,
    CycleResult,
    ExecutionResult,
    JevJudgment,
    MarketSnapshot,
    PolicyDecision,
    TradeIntent,
)
from jev_trader.policy import apply_policy
from jev_trader.risk import (
    EntryGuardConfig,
    EntryGuardState,
    apply_risk,
    build_entry_guard_state,
    daily_loss_hit,
    entry_guard_skip_reason,
    load_entry_guard_config,
    long_protective_stop,
    loss_streak_from_closes,
)
from jev_trader.state import build_compact_state
from jev_trader.status import log_event
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
            "action_probabilities": dict(result.judgment.action_probabilities or {}),
            "model": result.judgment.model,
        },
        "state_text": result.state_text,
    }
    if result.judge_ms is not None:
        payload["judge_ms"] = round(float(result.judge_ms), 1)
    if result.model_skipped:
        payload["model_skipped"] = True
    return payload


# Label on the synthetic judgment when the model was never asked.
MODEL_SKIPPED = "model_skipped"


def pre_model_entry_block(
    snapshot: MarketSnapshot,
    account: AccountState,
    config: EntryGuardConfig,
    state: EntryGuardState,
    *,
    now: datetime,
) -> str | None:
    """Guard that blocks any entry for a flat book, so the model call is wasted.

    Order: wallet_unavailable (no wallet this cycle), kill_switch, daily_loss, then
    entry_guard_skip_reason (no_entry_window, hourly_entry_cap,
    loss_streak_pause). An open position returns None (the model still
    decides closes; kill_switch there is a MARKET close from apply_risk).
    """
    in_position = snapshot.position.side != "FLAT" and snapshot.position.size > 0
    if in_position:
        return None
    if not account.wallet_ok:
        return "wallet_unavailable"
    if account.kill_switch:
        return "kill_switch"
    if daily_loss_hit(account.daily_pnl_pct, account.daily_loss_limit_pct):
        return "daily_loss"
    return entry_guard_skip_reason(config, state, now=now)


def model_skipped_judgment(reason: str) -> JevJudgment:
    """Hold stand-in recorded when the model was not called."""
    return JevJudgment(
        action="hold",
        trend_aligned=0.0,
        false_break_risk=0.0,
        signal_strength="skipped",
        should_trade_now=0.0,
        action_probabilities={},
        model=MODEL_SKIPPED,
        raw={MODEL_SKIPPED: True, "reason": reason},
    )


def model_skipped_intent(
    snapshot: MarketSnapshot, account: AccountState, reason: str
) -> TradeIntent:
    """Same shape as apply_risk's skip(): hold, qty 0, risk_event = reason."""
    return TradeIntent(
        action="hold",
        qty=0.0,
        stop_price=None,
        stop_distance=None,
        entry_type="NONE",
        reduce_only=False,
        client_order_id=client_order_id(snapshot.symbol, "hold"),
        symbol=snapshot.symbol,
        risk_pct=account.risk_pct,
        order_side="BUY",
        skip_reason=reason,
        risk_event=reason,
    )


def arm_exchange_stop(
    broker: Any,
    ledger: Ledger | None,
    symbol: str,
    stop_price: float | None,
    detail: dict[str, Any] | None,
) -> Any:
    """Remember the stop and make sure the exchange has a close-all STOP_MARKET.

    Returns the broker's answer (None when there is nothing to arm).
    """
    if not symbol or stop_price is None or float(stop_price) <= 0:
        return None
    price = float(stop_price)
    with order_lock:
        if ledger is not None:
            try:
                ledger.set_stop(symbol, price)
            except Exception as exc:  # noqa: BLE001 — a missing ledger row must not block the order
                log_event("ledger_stop_error", symbol=symbol, error=f"{type(exc).__name__}: {exc}")
        ensure = getattr(broker, "ensure_stop_market", None)
        if not callable(ensure):
            return None
        try:
            res = ensure(symbol, stop_price=price, order_side="SELL")
        except Exception as exc:  # noqa: BLE001 — fill already happened; surface the miss
            res = {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(res, dict) and res.get("error"):
            log_event("stop_error", symbol=symbol, stop_price=price, error=res["error"])
        if isinstance(detail, dict):
            if isinstance(res, dict) and res.get("error"):
                detail["stop_error"] = res["error"]
            else:
                detail["stop"] = res
        return res


def recorded_outcome(
    policy: PolicyDecision,
    intent: TradeIntent,
    execution: ExecutionResult | None,
) -> tuple[str, str | None]:
    """What the ledger should call this cycle.

    A filled risk close is the trade even when the model's close failed policy.
    Otherwise policy wins, then the order status (unfilled / working).
    """
    if (
        intent.action == "close"
        and intent.risk_event is not None
        and is_real_fill(execution)
        and fill_qty(execution, intent.qty) > 0
    ):
        return "close", None
    if not policy.passed:
        action, skip = "hold", policy.skip_reason
    elif intent.skip_reason:
        action, skip = "hold", intent.skip_reason
    else:
        action, skip = intent.action, None
    status = None if execution is None else execution.status
    if status == "unfilled":
        # Risk exits that never filled should not surface a policy skip reason.
        if intent.risk_event is not None:
            return "hold", "unfilled"
        return "hold", skip or "unfilled"
    if status == "working":
        return action, skip or "working"
    if status == "submit_unknown":
        # The reconciler finds out by clientOrderId and records any fill.
        return "hold", "submit_unknown"
    return action, skip


def run_once(
    snapshot: MarketSnapshot,
    *,
    judgment: JevJudgment | None = None,
    jev_client: Any | None = None,  # JudgeClient protocol (JevClient | LayaClient)
    account: AccountState | None = None,
    broker: PaperBroker | BinanceTestnetBroker | BinanceFuturesBroker | None = None,
    ledger: Ledger | None = None,
    notifier: TelegramNotifier | None = None,
    typesafe_api_key: str | None = None,
    follow_jev: bool = False,
    min_should_trade: float | None = None,
    min_hold_sec: float | None = None,
    entry_guard_config: EntryGuardConfig | None = None,
    now: datetime | None = None,
) -> CycleResult:
    """One paper/testnet decision cycle: features → state → Jev → policy → risk → exec.

    When the book is flat and an entry guard is already active, the model is not
    called: the cycle records a hold with that guard as skip_reason.
    """
    features = compute_features(snapshot)
    compact = build_compact_state(snapshot, features)
    exec_broker = broker or PaperBroker()
    acct = account or AccountState(
        equity_usdt=snapshot.position.cash_usdt,
        daily_pnl_pct=0.0,
        kill_switch=False,
        open_positions=0 if snapshot.position.side == "FLAT" else 1,
    )
    seen_flatten = current_flatten_generation()
    protective = long_protective_stop(snapshot, features, acct)
    if protective is not None and protective < features.close:
        arm_exchange_stop(exec_broker, ledger, snapshot.symbol, protective, None)
    # Guard inputs come first so a blocked flat book never pays for a model call.
    now = now or datetime.now(timezone.utc)
    if entry_guard_config is None:
        entry_guard_config = load_entry_guard_config()
    seconds_since_close = None
    seconds_since_entry = None
    entries_last_hour = 0
    loss_streak = 0
    last_loss_ts = None
    if ledger is not None:
        seconds_since_close = ledger.seconds_since_last_close(snapshot.symbol)
        seconds_since_entry = ledger.seconds_since_last_entry(snapshot.symbol)
        entries_last_hour = ledger.count_entries_since(now - timedelta(hours=1))
        loss_streak, last_loss_ts = loss_streak_from_closes(
            ledger.recent_close_pnls(limit=64)
        )
    entry_guard_state = build_entry_guard_state(
        config=entry_guard_config,
        now=now,
        entries_last_hour=entries_last_hour,
        loss_streak=loss_streak,
        last_loss_ts=last_loss_ts,
    )

    resolved = judgment
    judge_ms: float | None = None
    pre_skip: str | None = None
    if resolved is None:
        pre_skip = pre_model_entry_block(
            snapshot, acct, entry_guard_config, entry_guard_state, now=now
        )

    if pre_skip is not None:
        resolved = model_skipped_judgment(pre_skip)
        policy = PolicyDecision(
            action="hold", passed=False, skip_reason=pre_skip, judgment=resolved
        )
        intent = model_skipped_intent(snapshot, acct, pre_skip)
    else:
        if resolved is None:
            client = jev_client
            owns = False
            if client is None:
                if not typesafe_api_key:
                    raise ValueError("Jev judgment missing and TYPESAFE_API_KEY not provided")
                client = JevClient(api_key=typesafe_api_key)
                owns = True
            try:
                t0 = time.perf_counter()
                resolved = client.judge(compact)
                judge_ms = (time.perf_counter() - t0) * 1000.0
            finally:
                if owns:
                    client.close()

        policy_kwargs: dict[str, Any] = {"follow_jev": follow_jev}
        if min_should_trade is not None:
            policy_kwargs["min_should_trade"] = min_should_trade
        policy = apply_policy(resolved, **policy_kwargs)
        risk_kwargs: dict[str, Any] = {
            "seconds_since_last_close": seconds_since_close,
            "seconds_since_last_entry": seconds_since_entry,
            "entry_guard_config": entry_guard_config,
            "entry_guard_state": entry_guard_state,
            "now": now,
        }
        if min_hold_sec is not None:
            risk_kwargs["min_hold_sec"] = min_hold_sec
        intent = apply_risk(
            policy,
            features,
            snapshot,
            acct,
            **risk_kwargs,
        )
    with order_lock:
        if (
            intent.action == "buy_long"
            and intent.skip_reason is None
            and current_flatten_generation() != seen_flatten
        ):
            intent = replace(intent, skip_reason="flatten", qty=0.0, risk_event="flatten")
        execution = exec_broker.submit(intent)
        if (
            is_real_fill(execution)
            and intent.action == "buy_long"
            and intent.stop_price
            and fill_qty(execution, intent.qty) > 0
            and float(intent.stop_price) < features.close
        ):
            arm_exchange_stop(
                exec_broker,
                ledger,
                intent.symbol,
                float(intent.stop_price),
                execution.detail if isinstance(execution.detail, dict) else None,
            )

    action, skip_reason = recorded_outcome(policy, intent, execution)

    result = CycleResult(
        action=action,
        skip_reason=skip_reason,
        intent=intent,
        execution=execution,
        judgment=resolved,
        state_text=compact.as_text(),
        state=compact.as_dict(),
        risk_event=intent.risk_event,
        judge_ms=judge_ms,
        model_skipped=pre_skip is not None,
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
