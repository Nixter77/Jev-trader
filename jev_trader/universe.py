"""Pick a trade list from the full UM ticker universe.

The bot *sees* every listed USD-M symbol. Jev is only asked about a short
liquid list plus already-open positions — not 700+ calls every 5 minutes.
"""

from __future__ import annotations

from collections.abc import Iterable

from jev_trader.public_market import UniverseTicker

DEFAULT_UNIVERSE_SIZE = 15
DEFAULT_MIN_QUOTE_VOLUME = 5_000_000.0
HARD_CAP = 40


def is_usdt_perp(symbol: str) -> bool:
    """Linear USDT perpetual, not a dated contract or USDC pair."""
    s = (symbol or "").upper()
    if not s.endswith("USDT"):
        return False
    if "_" in s:
        return False
    return True


def select_trade_universe(
    rows: Iterable[UniverseTicker],
    *,
    limit: int = DEFAULT_UNIVERSE_SIZE,
    min_quote_volume: float = DEFAULT_MIN_QUOTE_VOLUME,
    extra: Iterable[str] = (),
) -> list[str]:
    """Open positions first, then highest 24h quote-volume USDT perps."""
    cap = min(HARD_CAP, max(1, int(limit)))
    extras: list[str] = []
    seen: set[str] = set()
    for raw in extra:
        symbol = str(raw or "").upper()
        if not symbol or symbol in seen:
            continue
        extras.append(symbol)
        seen.add(symbol)

    ranked = [
        row
        for row in rows
        if is_usdt_perp(row.symbol) and float(row.quote_volume or 0.0) >= float(min_quote_volume)
    ]
    ranked.sort(key=lambda row: float(row.quote_volume or 0.0), reverse=True)

    chosen = list(extras)
    for row in ranked:
        symbol = row.symbol.upper()
        if symbol in seen:
            continue
        if len(chosen) >= cap:
            break
        chosen.append(symbol)
        seen.add(symbol)
    return chosen
