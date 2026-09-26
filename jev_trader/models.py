from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Action = Literal["buy_long", "sell_short", "close", "hold"]
PositionSide = Literal["FLAT", "LONG", "SHORT"]
# Flat account is blocked when day PnL / day-start equity is at or under this.
DAILY_LOSS_LIMIT_PCT = 0.025
SIGNAL_STRENGTH_LEVELS: tuple[str, ...] = ("нет края", "слабый", "рабочий", "сильный")
ACTIONS: tuple[str, ...] = ("buy_long", "sell_short", "close", "hold")


@dataclass(frozen=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class BookLevel:
    price: float
    qty: float


@dataclass(frozen=True)
class OrderBook:
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]


@dataclass(frozen=True)
class Position:
    side: PositionSide
    size: float
    cash_usdt: float
    entry: float | None = None
    upnl_pct: float | None = None
    bars_in_trade: int = 0
    stop_price: float | None = None


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    tf: str
    candles: tuple[Candle, ...]
    position: Position
    book: OrderBook | None = None
    news: tuple[str, ...] = ()
    funding: float | None = None
    doi_1h: float | None = None
    btc_corr: float | None = None


@dataclass(frozen=True)
class Features:
    close: float
    ret_1: float | None
    ret_5: float | None
    ret_12: float | None
    atr: float | None
    atr_pct: float | None
    bar_range_atr: float | None
    ema20: float | None
    ema50: float | None
    ema200: float | None
    ema_stack: str
    adx: float | None
    rsi: float | None
    vwap: float | None
    vwap_dist_pct: float | None
    structure: str
    last_swing_low_pct: float | None
    last_swing_high_pct: float | None
    book_imbalance: float | None
    spread_bps: float | None
    ts: int


@dataclass(frozen=True)
class CompactState:
    payload: dict[str, Any]
    text: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def as_text(self) -> str:
        return self.text


@dataclass(frozen=True)
class JevJudgment:
    action: str
    trend_aligned: float
    false_break_risk: float
    signal_strength: str
    should_trade_now: float
    action_probabilities: dict[str, float] = field(default_factory=dict)
    signal_strength_score: float | None = None
    model: str = "jev-1.13.0"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    passed: bool
    skip_reason: str | None
    judgment: JevJudgment


@dataclass(frozen=True)
class TradeIntent:
    action: str
    qty: float
    stop_price: float | None
    stop_distance: float | None
    entry_type: str
    reduce_only: bool
    client_order_id: str
    symbol: str
    risk_pct: float
    order_side: str = "BUY"
    limit_price: float | None = None
    skip_reason: str | None = None
    risk_event: str | None = None


def market_close_intent(
    *,
    symbol: str,
    qty: float,
    order_side: str,
    client_order_id: str,
    risk_event: str | None,
    risk_pct: float = 0.0,
    limit_price: float | None = None,
    entry_type: str = "MARKET",
) -> TradeIntent:
    """Reduce-only close. Every caller (risk, flatten, exchange) builds the same order."""
    return TradeIntent(
        action="close",
        qty=qty,
        stop_price=None,
        stop_distance=None,
        entry_type=entry_type,
        reduce_only=True,
        client_order_id=client_order_id,
        symbol=symbol,
        risk_pct=risk_pct,
        order_side=order_side,
        limit_price=limit_price,
        skip_reason=None,
        risk_event=risk_event,
    )


@dataclass(frozen=True)
class AccountState:
    equity_usdt: float
    daily_pnl_pct: float
    kill_switch: bool
    open_positions: int
    max_positions: int = 3
    risk_pct: float = 0.005
    atr_stop_mult: float = 1.5
    daily_loss_limit_pct: float = DAILY_LOSS_LIMIT_PCT
    leverage: float = 3.0
    available_usdt: float | None = None


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    venue: str
    client_order_id: str
    reduce_only: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CycleResult:
    action: str
    skip_reason: str | None
    intent: TradeIntent | None
    execution: ExecutionResult | None
    judgment: JevJudgment | None
    state_text: str
    state: dict[str, Any]
    risk_event: str | None = None
    judge_ms: float | None = None
    model_skipped: bool = False
