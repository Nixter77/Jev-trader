# Laya close bleed — plan + tip 1

**Date:** 2026-09-22 Asia/Jerusalem  
**Branch:** `feat/laya-close-asymmetry`  
**Evidence (ledger-testnet):** since `2026-09-21T23:28:51Z` — 18 closes, sum ≈ −56 USDT (16 neg / 2 pos). Pre last30 closes ≈ +174. All closes `model=laya:typed-decisions`.

## Root cause (code + ledger)
`desk` defaults `--follow-jev` → `follow_jev=True` skips **all** policy probability gates. Laya close rows had `should_trade_now` **0.35–0.50** (med ~0.44) and `close` prob **0.57–0.70**. None would pass entry bar 0.72; they still closed.

## Plan (ordered)
1. **FIRST (this tip) — asymmetry (option 2):** even under `follow_jev`, **close always gated**; `CLOSE_SHOULD_TRADE_MIN=0.80` (> buy 0.72); if `action_probabilities` present, `close` prob ≥ 0.75. Skip reasons: `close_should_trade_now` / `close_action_prob`. Exchange stop / kill / daily_loss via `risk.py` unchanged.
2. **Next — min hold (option 1):** block discretionary close for N minutes/bars after entry (stop still allowed).
3. **Then — MFE guard (option 3):** no discretionary close if unrealized R > X unless signal catastrophic.
4. **Safety — kill-switch (option 4):** N consecutive losing discretionary closes → pause Laya closes or fallback Jev.

**Not doing:** “just switch backend to Jev” as the fix.

## Counterfactual
All 18 post-cut closes fail `should_trade_now >= 0.80` and almost all fail `close_prob >= 0.75` → would hold.

## edge?
no claim — gate only. Live restart needs Chief GO.

---

## Tip 2026-09-23 — backend-aware asymmetry (shipped)

**Evidence (ledger-testnet closes, discretionary only):**
- Gate soft Laya closes (`should < 0.80` or `close_p < 0.75`): would block 24 fills, sum ≈ **−62 USDT** (good).
- Same gate on Jev: would block 107 fills, sum ≈ **+539 USDT** (destroys the desk exit).

**Code:** `policy.apply_policy` — under `follow_jev`, Jev soft-closes; Laya close stays gated. `risk.MIN_HOLD_SEC=900` blocks discretionary close until entry ages; stop/kill/daily unchanged.

**Ops:** restart live/`desk` to load code. Prefer `DECISION_BACKEND=jev` until Laya close quality improves; Laya is safe to A/B with this gate.

## Tip 2026-09-23b — trial lower Laya close floors (multi-hour A/B)

Live after Jev→Laya switch: Laya judgment ~80% buy / 0% hold; all closes blocked at 0.80 → 0 fills + max_positions lock.

**Trial floors** (Laya / strict gated close only; Jev under follow_jev unchanged):
- `CLOSE_SHOULD_TRADE_MIN = 0.45` (was 0.80)
- `CLOSE_ACTION_PROB_MIN = 0.55` (was 0.75)
- `CLOSE_FALSE_BREAK_RISK_MAX = 0.45` on gated close (entry quality still 0.35)

Counterfactual on post-switch Laya closes: ~16/45 would pass. Restart desk to load. Revisit after hourly compares.
