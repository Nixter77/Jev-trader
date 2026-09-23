"""Pick a trade list from the full UM ticker universe.

The bot *sees* every listed USD-M symbol. Jev is only asked about a short
liquid list plus already-open positions — not 700+ calls every 5 minutes.

Local Laya (~20s/symbol on CPU) cannot finish a 15-symbol pass inside one 5m
bar, so when DECISION_BACKEND=laya the default watch size shrinks to
DEFAULT_LAYA_UNIVERSE_SIZE (override with LAYA_UNIVERSE_SIZE or --universe-size).
Jev cloud stays on DEFAULT_UNIVERSE_SIZE.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

from jev_trader.public_market import UniverseTicker

DEFAULT_UNIVERSE_SIZE = 15
# Local Laya typed-decisions is ~22s/symbol on Intel CPU → 5 symbols ≈ 110s < 5m.
DEFAULT_LAYA_UNIVERSE_SIZE = 5
DEFAULT_MIN_QUOTE_VOLUME = 5_000_000.0
HARD_CAP = 40


def resolve_universe_size(
    *,
    backend: str = "jev",
    explicit: int | None = None,
    environ: dict[str, str] | None = None,
) -> int:
    """Pick watch-list size: explicit CLI wins, else Laya-aware default / env."""
    if explicit is not None:
        return min(HARD_CAP, max(1, int(explicit)))
    env = os.environ if environ is None else environ
    backend_n = (backend or "jev").strip().lower()
    if backend_n == "laya":
        raw = (env.get("LAYA_UNIVERSE_SIZE") or "").strip()
        if raw:
            return min(HARD_CAP, max(1, int(raw)))
        return DEFAULT_LAYA_UNIVERSE_SIZE
    return DEFAULT_UNIVERSE_SIZE


def order_symbols_open_first(
    symbols: Sequence[str],
    open_symbols: Iterable[str] = (),
) -> list[str]:
    """Stable reorder: open positions first (open list order), then the rest.

    Used each live poll pass so exits/management for open books are judged
    before scanning cold watch symbols — safer when a full pass is slow.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in open_symbols:
        symbol = str(raw or "").upper()
        if not symbol or symbol in seen:
            continue
        ordered.append(symbol)
        seen.add(symbol)
    for raw in symbols:
        symbol = str(raw or "").upper()
        if not symbol or symbol in seen:
            continue
        ordered.append(symbol)
        seen.add(symbol)
    return ordered


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
