# Local Laya speed vs 5m bar

Cloud Jev finishes a full watch pass in seconds. Local Laya `typed-decisions`
on Intel CPU is ~22–23s per symbol, so 15 symbols ≈ 5.5 min (worse than one 5m bar).

Mitigations in code (no bot restart applied by the agent — restart `run` yourself):

1. **Open-positions-first** each `poll_once` (`order_symbols_open_first`) so open books are judged before cold watch symbols.
2. **Smaller Laya universe default** (`DEFAULT_LAYA_UNIVERSE_SIZE=5`, env `LAYA_UNIVERSE_SIZE`, or `--universe-size`). Jev stays at 15.
3. **`judge_ms`** on decision payload / `bot-status.json` `last_decisions` for latency visibility.
4. **No thread-pool parallelization** — shared Laya/torch predict is not assumed thread-safe; correctness preferred.
5. **`LAYA_DEVICE=mps`** only helps Apple Silicon with MPS; Intel Mac stays CPU (leave `LAYA_DEVICE` empty).

Expected: 5 × ~22s ≈ 110s per pass (still slow, but fits inside 5m with margin for market I/O).
