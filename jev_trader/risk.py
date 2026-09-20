from __future__ import annotations

import time

from jev_trader.models import (
    AccountState,
    Features,
    MarketSnapshot,
    PolicyDecision,
    TradeIntent,
)

ATR_STOP_MULT_MIN = 1.2
ATR_STOP_MULT_MAX = 1.8


def _client_order_id(symbol: str, action: str) -> str:
    return f"jev1_{symbol}_{action}_{int(time.time() * 1000)}"


def _stop_price(close: float, stop_distance: float) -> float:
    return close - stop_distance


def _order_side(action: str, position_side: str) -> str:
    """Binance long-only: BUY opens a long, SELL only closes a long."""
    if action == "buy_long":
        return "BUY"
    if action == "close" and position_side == "SHORT":
        return "BUY"
    if action in {"close", "sell_short"}:
        return "SELL"
    return "BUY"


def post_only_limit_price(order_side: str, close: float, snapshot: MarketSnapshot) -> float:
    """Maker price: join the bid on BUY, the ask on SELL; else last close."""
    book = snapshot.book
    if book is not None:
        if order_side == "BUY" and book.bids:
            return book.bids[0].price
        if order_side == "SELL" and book.asks:
            return book.asks[0].price
    return close


def apply_risk(
    policy: PolicyDecision,
    features: Features,
    snapshot: MarketSnapshot,
    account: AccountState,
) -> TradeIntent:
    """Size = risk% / ATR-stop. Jev probabilities never enter the size formula."""
    symbol = snapshot.symbol
    close = features.close
    stop_mult = min(max(account.atr_stop_mult, ATR_STOP_MULT_MIN), ATR_STOP_MULT_MAX)

    def skip(reason: str, action: str = "hold") -> TradeIntent:
        return TradeIntent(
            action=action,
            qty=0.0,
            stop_price=None,
            stop_distance=None,
            entry_type="NONE",
            reduce_only=False,
            client_order_id=_client_order_id(symbol, action),
            symbol=symbol,
            risk_pct=account.risk_pct,
            order_side=_order_side(action, snapshot.position.side),
            skip_reason=reason,
            risk_event=reason,
        )

    in_position = snapshot.position.side != "FLAT" and snapshot.position.size > 0
    pos_side = snapshot.position.side

    def close_position(*, risk_event: str | None, entry_type: str) -> TradeIntent:
        side = _order_side("close", pos_side)
        limit = None if entry_type == "MARKET" else post_only_limit_price(side, close, snapshot)
        tag = "flatten" if entry_type == "MARKET" else "close"
        return TradeIntent(
            action="close",
            qty=snapshot.position.size,
            stop_price=None,
            stop_distance=None,
            entry_type=entry_type,
            reduce_only=True,
            client_order_id=_client_order_id(symbol, tag),
            symbol=symbol,
            risk_pct=account.risk_pct,
            order_side=side,
            limit_price=limit,
            skip_reason=None,
            risk_event=risk_event,
        )

    if account.kill_switch:
        if in_position:
            return close_position(risk_event="kill_switch", entry_type="MARKET")
        return skip("kill_switch")

    if account.daily_pnl_pct <= -abs(account.daily_loss_limit_pct):
        if in_position:
            return close_position(risk_event="daily_loss", entry_type="MARKET")
        return skip("daily_loss")

    # Long-only: leftover shorts are flattened; SELL never opens a position.
    if in_position and pos_side == "SHORT":
        return close_position(risk_event="no_short", entry_type="MARKET")

    if in_position and pos_side == "LONG":
        stop = snapshot.position.stop_price
        if stop is None and features.atr and features.atr > 0 and snapshot.position.entry:
            stop = snapshot.position.entry - stop_mult * features.atr
        if stop is not None and close <= stop:
            return close_position(risk_event="stop", entry_type="MARKET")

    if not policy.passed:
        return skip(policy.skip_reason or "hold")

    if policy.action == "sell_short":
        return skip("no_short")

    if policy.action == "close":
        if not in_position or pos_side != "LONG":
            return skip("flat")
        return close_position(risk_event=None, entry_type="MARKET")

    if policy.action != "buy_long":
        return skip(policy.skip_reason or "hold")

    if in_position and pos_side == "LONG":
        return skip("already_long")

    if account.open_positions >= account.max_positions:
        return skip("max_positions")

    atr = features.atr
    if atr is None or atr <= 0 or close <= 0:
        return skip("no_atr")

    stop_distance = stop_mult * atr
    risk_amount = account.equity_usdt * account.risk_pct
    qty = risk_amount / stop_distance
    available = account.available_usdt
    if available is None:
        available = account.equity_usdt
    leverage = max(float(account.leverage or 1.0), 1.0)
    max_notional = max(0.0, float(available)) * leverage
    if close > 0 and max_notional > 0:
        qty = min(qty, max_notional / close)
    if qty <= 0:
        return skip("insufficient_margin")
    return TradeIntent(
        action="buy_long",
        qty=qty,
        stop_price=_stop_price(close, stop_distance),
        stop_distance=stop_distance,
        entry_type="LIMIT_POST_ONLY",
        reduce_only=False,
        client_order_id=_client_order_id(symbol, "buy_long"),
        symbol=symbol,
        risk_pct=account.risk_pct,
        order_side="BUY",
        limit_price=post_only_limit_price("BUY", close, snapshot),
        skip_reason=None,
        risk_event=None,
    )
