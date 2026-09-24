# Entry guards (no-trade window, hourly cap, loss-streak pause)

Ledger analysis (testnet desk): closes in 03:00–09:00 Asia/Jerusalem lost ≈−184 USDT
over 5 days while other hours made ≈+288; pace spiked to ~3.6 closes/h in the first
hours after the UTC-midnight daily-loss reset; loss streaks of 10–12 then left the
bot blocked by `daily_loss` for the rest of the day.

These three guards block **new entries only** (`buy_long`). Discretionary closes,
exchange stops, kill-switch flatten, daily-loss flatten, and no-short flatten are
unchanged and still fire immediately.

## Defaults

| Guard | Env | Default | Skip reason | Disable |
|---|---|---|---|---|
| No-new-entry window | `NO_ENTRY_WINDOW`, `NO_ENTRY_TZ` | `03:00-09:00` / `Asia/Jerusalem` | `no_entry_window` | `NO_ENTRY_WINDOW=off` (or empty) |
| Max entries / rolling hour | `MAX_ENTRIES_PER_HOUR` | `2` | `hourly_entry_cap` | `MAX_ENTRIES_PER_HOUR=0` |
| Loss-streak pause | `LOSS_STREAK_PAUSE_N`, `LOSS_STREAK_PAUSE_MIN` | `3` / `120` min | `loss_streak_pause` | `LOSS_STREAK_PAUSE_N=0` |

Window format is `HH:MM-HH:MM` in the named IANA zone (DST via `zoneinfo`). Windows
that wrap midnight (e.g. `22:00-06:00`) are supported. Hourly cap counts `buy_long`
fills across all symbols from the ledger. Loss streak walks close fills’ 
`realized_pnl_usdt` newest-first; break-even (`>= 0`) resets the streak. Pause is
derived from the ledger (survives restarts); after the cooldown expires, entries
may resume even if the streak has not yet been broken by a winning close.

## Status

`bot-status.json` includes:

```json
"entry_guards": {
  "window_active": false,
  "entries_last_hour": 1,
  "loss_streak": 2,
  "pause_until": null
}
```

## Restart note

Guards load from process env / `.env` at cycle time. Restart the bot process after
changing them (do not kill a live pid from an agent session unless the operator asks).
