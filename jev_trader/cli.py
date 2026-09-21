from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from jev_trader.config import LIVE_CONFIRM_VALUE, load_settings
from jev_trader.cycle import dumps_decision, run_once
from jev_trader.execution import BinanceFuturesBroker, BinanceTestnetBroker, PaperBroker

DEFAULT_PAPER_LEDGER = "data/ledger.sqlite"
DEFAULT_TESTNET_LEDGER = "data/ledger-testnet.sqlite"
from jev_trader.universe import DEFAULT_MIN_QUOTE_VOLUME, DEFAULT_UNIVERSE_SIZE
from jev_trader.jev import JevClient, judgment_from_dict
from jev_trader.judge import make_judge_client, normalize_backend
from jev_trader.laya_client import DEFAULT_LAYA_CHECKPOINT
from jev_trader.ledger import Ledger
from jev_trader.live import LiveRunner, load_answers, load_json, make_run_cycle
from jev_trader.models import AccountState
from jev_trader.flatten import run_flatten
from jev_trader.monitor import bind_monitor, dashboard_state
from jev_trader.public_market import (
    compact_universe_payload,
    fetch_universe_summary,
    universe_payload,
)
from jev_trader.snapshot import load_snapshot
from jev_trader.state import build_compact_state
from jev_trader.status import clear_pid, other_bot_running, read_json, resolve_day_anchor, write_pid
from jev_trader.telegram import notifier_from_settings


def _ledger_path(args: argparse.Namespace) -> str:
    path = args.ledger
    if args.venue == "testnet" and path == DEFAULT_PAPER_LEDGER:
        return DEFAULT_TESTNET_LEDGER
    return path


def _require_binance_keys(settings, venue: str) -> str | None:
    if venue not in {"testnet", "live"}:
        return None
    if settings.binance_api_key and settings.binance_api_secret:
        return None
    return "BINANCE_API_KEY / BINANCE_API_SECRET missing in .env"


def _flatten_callback(broker, ledger: Ledger, status_path, runner=None):
    def _run() -> dict:
        return run_flatten(broker, ledger, status_path=status_path, runner=runner)

    return _run


def _flatten_callback_from_env(args: argparse.Namespace, ledger: Ledger, status_path):
    venue = getattr(args, "venue", None)
    if not venue:
        status = read_json(status_path) or {}
        venue = status.get("venue") or "paper"
    try:
        settings = load_settings(
            getattr(args, "env", ".env"),
            allow_production=venue == "live",
        )
        ns = argparse.Namespace(venue=venue)
        missing = _require_binance_keys(settings, venue)
        if missing:
            return None
        broker = _make_broker(ns, settings)
    except Exception:  # noqa: BLE001
        return None
    return _flatten_callback(broker, ledger, status_path)


def _make_broker(args: argparse.Namespace, settings) -> PaperBroker | BinanceTestnetBroker | BinanceFuturesBroker:
    if args.venue == "paper":
        return PaperBroker()
    if args.venue == "testnet":
        return BinanceTestnetBroker(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            base_url=settings.binance_fapi_base,
        )
    return BinanceFuturesBroker(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        base_url=settings.binance_fapi_base,
        live=True,
    )



def _apply_backend_args(settings, args):
    """Overlay --backend / --laya-* CLI flags onto Settings (frozen dataclass)."""
    from dataclasses import replace

    updates = {}
    backend = getattr(args, "backend", None)
    if backend:
        updates["decision_backend"] = normalize_backend(backend)
    checkpoint = getattr(args, "laya_checkpoint", None)
    if checkpoint:
        updates["laya_checkpoint"] = checkpoint
    device = getattr(args, "laya_device", None)
    if device:
        updates["laya_device"] = device
    return replace(settings, **updates) if updates else settings




def _judge_kwargs(settings, answers=None) -> dict:
    """Kwargs for run_once / live loop: Jev key or Laya client."""
    if answers is not None:
        return {"typesafe_api_key": None, "jev_client": None}
    from jev_trader.judge import normalize_backend, make_judge_client
    if normalize_backend(settings.decision_backend) == "laya":
        return {"typesafe_api_key": None, "jev_client": make_judge_client(settings)}
    return {"typesafe_api_key": settings.typesafe_api_key, "jev_client": None}

def _require_judge_ready(settings) -> str | None:
    """Return an error string if the selected backend cannot run, else None."""
    backend = normalize_backend(settings.decision_backend)
    if backend == "jev" and not settings.typesafe_api_key:
        return "TYPESAFE_API_KEY missing after env map (backend=jev)"
    return None


def _add_backend_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        choices=("jev", "laya"),
        default=None,
        help="Decision model: jev (TypeSafe API) or laya (local HF). Overrides DECISION_BACKEND.",
    )
    p.add_argument(
        "--laya-checkpoint",
        default=None,
        help="Laya checkpoint: typed-decisions (default), multilingual, english, router. Overrides LAYA_CHECKPOINT.",
    )
    p.add_argument(
        "--laya-device",
        default=None,
        help="Optional torch device for Laya (cpu/cuda/mps). Overrides LAYA_DEVICE.",
    )

def cmd_once(args: argparse.Namespace) -> int:
    settings = load_settings(args.env, allow_production=args.venue == "live")
    settings = _apply_backend_args(settings, args)
    snapshot = load_snapshot(args.snapshot)
    answers = load_answers(args.answers)
    missing = _require_binance_keys(settings, args.venue)
    if missing:
        print(json.dumps({"ok": False, "error": missing}))
        return 2
    ledger = Ledger(_ledger_path(args))
    broker = _make_broker(args, settings)
    equity = snapshot.position.cash_usdt
    open_n = 0 if snapshot.position.side == "FLAT" else 1
    fetch = getattr(broker, "fetch_wallet", None)
    if callable(fetch):
        wallet = fetch()
        equity = float(wallet.get("equity_usdt") or equity)
        open_n = int(wallet.get("open_positions") or open_n)
    account = AccountState(
        equity_usdt=equity,
        daily_pnl_pct=args.daily_pnl_pct,
        kill_switch=args.kill_switch,
        open_positions=open_n,
    )
    notifier = notifier_from_settings(settings, force_off=args.no_telegram)
    result = run_once(
        snapshot,
        judgment=None if answers is None else judgment_from_dict(answers),
        account=account,
        broker=broker,
        ledger=ledger,
        notifier=notifier,
        typesafe_api_key=settings.typesafe_api_key if settings.decision_backend == "jev" else None,
        jev_client=None if answers is not None or settings.decision_backend == "jev" else make_judge_client(settings),
    )
    print(dumps_decision(result))
    return 0


def cmd_jev_live(args: argparse.Namespace) -> int:
    settings = _apply_backend_args(load_settings(args.env), args)
    err = _require_judge_ready(settings)
    if err:
        print(json.dumps({"ok": False, "error": err}))
        return 2
    snapshot = load_snapshot(args.snapshot)
    compact = build_compact_state(snapshot)
    try:
        with make_judge_client(settings) as client:
            payload_model = client.model
            judgment = client.judge(compact)
    except Exception as exc:  # noqa: BLE001 — live probe must capture transport/API failures
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "model": "jev-1.13.0",
                },
                ensure_ascii=False,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "model": judgment.model,
                "client_model": payload_model,
                "action": judgment.action,
                "trend_aligned": judgment.trend_aligned,
                "false_break_risk": judgment.false_break_risk,
                "signal_strength": judgment.signal_strength,
                "should_trade_now": judgment.should_trade_now,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """All-symbol UM 24h ticker via unsigned public REST. Does not load API keys onto the wire."""
    interval = max(1.0, float(args.interval))
    started = time.monotonic()
    n = 0
    try:
        while True:
            try:
                rows = fetch_universe_summary()
            except Exception as exc:  # noqa: BLE001
                print(
                    json.dumps(
                        {"ok": False, "type": "universe", "error": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                    )
                )
                return 1
            payload = universe_payload(rows) if args.full else compact_universe_payload(rows)
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            n += 1
            if args.once:
                return 0
            if args.max_cycles is not None and n >= args.max_cycles:
                return 0
            if args.max_runtime is not None and time.monotonic() - started >= args.max_runtime:
                return 0
            sleep_for = interval
            if args.max_runtime is not None:
                remaining = args.max_runtime - (time.monotonic() - started)
                if remaining <= 0:
                    return 0
                sleep_for = min(sleep_for, remaining)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print(
            json.dumps(
                {"ok": True, "stopped": "interrupt", "type": "universe", "cycles": n},
                ensure_ascii=False,
            )
        )
        return 0


def _parse_symbols(raw: str | None, fallback: str) -> list[str]:
    text = (raw or fallback or "BTCUSDT").strip()
    symbols = [part.strip().upper() for part in text.split(",") if part.strip()]
    return symbols or ["BTCUSDT"]


def _sidecar(ledger: Ledger, name: str, override: str | None = None) -> Path:
    if override:
        return Path(override)
    return ledger.path.parent / name


def _make_live_runner(args: argparse.Namespace, settings, ledger: Ledger, broker, notifier) -> LiveRunner:
    answers = load_answers(getattr(args, "answers", None))
    judgment = None if answers is None else judgment_from_dict(answers)
    follow_jev = bool(getattr(args, "follow_jev", False))
    wallet_box: dict = {}
    fetch = getattr(broker, "fetch_wallet", None)
    cancel_entries = getattr(broker, "cancel_working_entries", None)
    if callable(fetch):
        try:
            # Drop leftover post-only entries. Keep STOP_MARKET so a restart
            # does not leave an open long unprotected until the next bar.
            if callable(cancel_entries):
                cancel_entries()
            wallet = fetch(ttl=0)
            wallet_box["wallet"] = wallet
            if wallet.get("equity_usdt"):
                equity = float(wallet["equity_usdt"])
                wallet_box["start_equity_usdt"] = resolve_day_anchor(
                    _sidecar(ledger, "risk-anchor.json"),
                    venue=str(args.venue),
                    equity=equity,
                )
            ledger.sync_exchange_positions(wallet)
        except Exception as exc:  # noqa: BLE001
            wallet_box["wallet_error"] = f"{type(exc).__name__}: {exc}"
    judge_kw = _judge_kwargs(settings, answers)
    run_cycle = make_run_cycle(
        judgment=judgment,
        account_kwargs={
            "daily_pnl_pct": args.daily_pnl_pct,
            "kill_switch": args.kill_switch,
            "max_positions": args.max_positions,
        },
        broker=broker,
        ledger=ledger,
        notifier=notifier,
        typesafe_api_key=judge_kw["typesafe_api_key"],
        jev_client=judge_kw["jev_client"],
        follow_jev=follow_jev,
        min_should_trade=args.min_should_trade,
        max_positions=args.max_positions,
        wallet_box=wallet_box,
    )
    recorded_klines = load_json(args.recorded_klines) if getattr(args, "recorded_klines", None) else None
    recorded_closes = [load_json(path) for path in (getattr(args, "recorded_close", None) or [])]
    recorded_depth = load_json(args.recorded_depth) if getattr(args, "recorded_depth", None) else None
    universe = bool(args.universe) and recorded_klines is None and not recorded_closes
    status_path = _sidecar(ledger, "bot-status.json", getattr(args, "status", None))
    return LiveRunner(
        _parse_symbols(getattr(args, "symbols", None), getattr(args, "symbol", "BTCUSDT")),
        run_cycle=run_cycle,
        kline_poll=args.kline_poll,
        summary_interval=args.summary_interval,
        max_cycles=args.max_cycles,
        max_runtime=args.max_runtime,
        print_universe=getattr(args, "print_universe", False),
        recorded_klines=recorded_klines,
        recorded_closes=recorded_closes,
        recorded_depth=recorded_depth,
        fire_latest=args.fire_latest,
        ledger=ledger,
        universe=universe,
        universe_size=args.universe_size,
        min_quote_volume=args.min_quote_volume,
        status_path=status_path,
        venue=args.venue,
        follow_jev=follow_jev,
        wallet_box=wallet_box,
    )


def cmd_live(args: argparse.Namespace) -> int:
    settings = load_settings(args.env, allow_production=args.venue == "live")
    settings = _apply_backend_args(settings, args)
    answers = load_answers(args.answers)
    settings = _apply_backend_args(settings, args)
    if answers is None:
        err = _require_judge_ready(settings)
        if err:
            print(json.dumps({"ok": False, "error": err}))
            return 2
    missing = _require_binance_keys(settings, args.venue)
    if missing:
        print(json.dumps({"ok": False, "error": missing}))
        return 2
    if args.venue == "live":
        print(
            json.dumps(
                {
                    "warning": "live venue sends SIGNED orders to Binance production",
                    "confirm": LIVE_CONFIRM_VALUE,
                    "base": settings.binance_fapi_base,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    ledger = Ledger(_ledger_path(args))
    broker = _make_broker(args, settings)
    notifier = notifier_from_settings(settings, force_off=args.no_telegram)
    runner = _make_live_runner(args, settings, ledger, broker, notifier)
    if args.venue in {"testnet", "live"} and runner.wallet_box.get("wallet_error"):
        print(json.dumps({"ok": False, "error": runner.wallet_box["wallet_error"]}))
        return 1
    try:
        return runner.run()
    except KeyboardInterrupt:
        print(
            json.dumps(
                {"ok": True, "stopped": "interrupt", "cycles": runner.cycles},
                ensure_ascii=False,
            )
        )
        return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    status_path = _sidecar(ledger, "bot-status.json", args.status)
    pid_path = _sidecar(ledger, "bot.pid", args.pid)
    flatten_fn = _flatten_callback_from_env(args, ledger, status_path)
    if args.json:
        print(json.dumps(dashboard_state(ledger, status_path=status_path, pid_path=pid_path, can_flatten=flatten_fn is not None), ensure_ascii=False, indent=2, default=str))
        return 0
    host, port = args.host, int(args.port)
    server = bind_monitor(
        ledger,
        host=host,
        port=port,
        status_path=status_path,
        pid_path=pid_path,
        flatten_fn=flatten_fn,
    )
    bound = server.server_address[1]
    print(
        json.dumps(
            {"ok": True, "monitor": f"http://{host}:{bound}", "ledger": str(ledger.path)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        return 0
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Always-on paper/testnet loop + local trade blotter."""
    settings = load_settings(args.env, allow_production=args.venue == "live")
    settings = _apply_backend_args(settings, args)
    answers = load_answers(args.answers)
    settings = _apply_backend_args(settings, args)
    if answers is None:
        err = _require_judge_ready(settings)
        if err:
            print(json.dumps({"ok": False, "error": err}))
            return 2
    missing = _require_binance_keys(settings, args.venue)
    if missing:
        print(json.dumps({"ok": False, "error": missing}))
        return 2
    if args.venue == "live":
        print(
            json.dumps(
                {
                    "warning": "live venue sends SIGNED orders to Binance production",
                    "confirm": LIVE_CONFIRM_VALUE,
                    "base": settings.binance_fapi_base,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    ledger = Ledger(_ledger_path(args))
    status_path = _sidecar(ledger, "bot-status.json", args.status)
    pid_path = _sidecar(ledger, "bot.pid", args.pid)
    host, port = args.host, int(args.port)
    existing = other_bot_running(pid_path)
    if existing is not None:
        print(
            json.dumps(
                {
                    "ok": True,
                    "already_running": True,
                    "pid": existing,
                    "monitor": f"http://{host}:{port}",
                    "hint": "бот уже жив; открываю только монитор",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        server = bind_monitor(
            ledger,
            host=host,
            port=port,
            status_path=status_path,
            pid_path=pid_path,
            flatten_fn=_flatten_callback_from_env(args, ledger, status_path),
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
        return 0

    broker = _make_broker(args, settings)
    notifier = notifier_from_settings(settings, force_off=args.no_telegram)
    try:
        runner = _make_live_runner(args, settings, ledger, broker, notifier)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    if args.venue in {"testnet", "live"} and runner.wallet_box.get("wallet_error"):
        print(json.dumps({"ok": False, "error": runner.wallet_box["wallet_error"]}))
        return 1
    stop = threading.Event()
    bounded = args.max_cycles is not None or args.max_runtime is not None
    bot_error: list[str] = []

    def bot_loop() -> None:
        while not stop.is_set():
            try:
                code = runner.run()
                if bounded or stop.is_set():
                    if code:
                        bot_error.append(f"bot exit {code}")
                    return
                time.sleep(3)
            except Exception as exc:  # noqa: BLE001
                runner.write_status(last_error=f"{type(exc).__name__}: {exc}")
                if bounded:
                    bot_error.append(f"{type(exc).__name__}: {exc}")
                    return
                if stop.wait(5):
                    return

    write_pid(pid_path)
    thread = threading.Thread(target=bot_loop, name="jev-live", daemon=True)
    thread.start()
    try:
        server = bind_monitor(
            ledger,
            host=host,
            port=port,
            status_path=status_path,
            pid_path=pid_path,
            flatten_fn=_flatten_callback(broker, ledger, status_path, runner=runner),
        )
    except OSError as exc:
        print(
            json.dumps(
                {
                    "ok": True,
                    "bot_running": True,
                    "monitor_error": f"{type(exc).__name__}: {exc}",
                    "hint": "бот уже крутится; откройте монитор на другом --port или существующий http://127.0.0.1:8787",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            while thread.is_alive() and not stop.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            stop.set()
        clear_pid(pid_path)
        return 0
    bound = server.server_address[1]
    print(
        json.dumps(
            {
                "ok": True,
                "monitor": f"http://{host}:{bound}",
                "venue": args.venue,
                "follow_jev": bool(args.follow_jev),
                "ledger": str(ledger.path),
                "wallet": None
                if not runner.wallet_box.get("wallet")
                else {
                    "equity_usdt": runner.wallet_box["wallet"].get("equity_usdt"),
                    "available_usdt": runner.wallet_box["wallet"].get("available_usdt"),
                    "open_positions": runner.wallet_box["wallet"].get("open_positions"),
                    "venue": runner.wallet_box["wallet"].get("venue"),
                },
                "hint": "оставьте процесс жить. Ctrl+C останавливает бота и монитор.",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop.set()
        server.shutdown()
        print(json.dumps({"ok": True, "stopped": "interrupt", "cycles": runner.cycles}, ensure_ascii=False))
        return 0
    finally:
        stop.set()
        clear_pid(pid_path)
    if bot_error:
        print(json.dumps({"ok": False, "error": bot_error[-1]}, ensure_ascii=False))
        return 1
    return 0


def cmd_trades(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    payload = ledger.book(limit=args.limit)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_flatten(args: argparse.Namespace) -> int:
    """Cancel open orders and MARKET-close every Binance position."""
    settings = load_settings(args.env, allow_production=args.venue == "live")
    missing = _require_binance_keys(settings, args.venue)
    if missing:
        print(json.dumps({"ok": False, "error": missing}))
        return 2
    if args.venue == "paper":
        print(json.dumps({"ok": False, "error": "flatten is for --venue testnet or live"}))
        return 2
    ledger = Ledger(_ledger_path(args))
    pid_path = _sidecar(ledger, "bot.pid", getattr(args, "pid", None))
    existing = other_bot_running(pid_path)
    if existing is not None and not args.force:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "bot is still running; Ctrl+C it first, or pass --force",
                    "pid": existing,
                }
            )
        )
        return 2
    if args.venue == "live":
        print(
            json.dumps(
                {
                    "warning": "flatten sends SIGNED MARKET closes to Binance production",
                    "confirm": LIVE_CONFIRM_VALUE,
                    "base": settings.binance_fapi_base,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    broker = _make_broker(args, settings)
    try:
        result = run_flatten(
            broker,
            ledger,
            status_path=_sidecar(ledger, "bot-status.json", None),
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    wallet = result.get("wallet") if isinstance(result, dict) else None
    print(
        json.dumps(
            {
                "ok": True,
                "venue": args.venue,
                "closes": None if not isinstance(result, dict) else result.get("closes"),
                "wallet": None
                if not isinstance(wallet, dict)
                else {
                    "equity_usdt": wallet.get("equity_usdt"),
                    "available_usdt": wallet.get("available_usdt"),
                    "open_positions": wallet.get("open_positions"),
                    "positions": wallet.get("positions"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_binance_ping(args: argparse.Namespace) -> int:
    settings = load_settings(args.env)
    try:
        broker = BinanceTestnetBroker(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            base_url=settings.binance_fapi_base,
        )
        result = broker.ping()
        result["ok"] = result.get("http_status") == 200
        if result["ok"]:
            try:
                wallet = broker.fetch_wallet()
                result["equity_usdt"] = wallet.get("equity_usdt")
                result["available_usdt"] = wallet.get("available_usdt")
                result["open_positions"] = wallet.get("open_positions")
            except Exception as exc:  # noqa: BLE001
                result["wallet_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Write commands on ONE line. In zsh a space after \\ turns the next\n"
            "flag (--snapshot, --env, --ledger, --symbol) into a new command.\n"
            "\n"
            "  python -m jev_trader run --venue testnet --env .env --no-telegram\n"
            "  python -m jev_trader flatten --venue testnet --env .env\n"
            "  python -m jev_trader run --venue paper --env .env --ledger data/ledger.sqlite --no-telegram\n"
            "  python -m jev_trader monitor --ledger data/ledger-testnet.sqlite\n"
            "  python -m jev_trader trades --ledger data/ledger.sqlite\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    once = sub.add_parser("once", help="Run one paper/testnet decision cycle")
    once.add_argument("--snapshot", required=True, help="Path to market snapshot JSON")
    once.add_argument("--answers", default=None, help="Optional replayed Jev answers JSON")
    once.add_argument("--env", default=".env")
    once.add_argument("--ledger", default="data/ledger.sqlite")
    once.add_argument("--venue", choices=("paper", "testnet", "live"), default="paper")
    once.add_argument("--kill-switch", action="store_true")
    once.add_argument("--daily-pnl-pct", type=float, default=0.0)
    once.add_argument("--no-telegram", action="store_true")
    _add_backend_flags(once)
    once.set_defaults(func=cmd_once)

    live = sub.add_parser("jev-live", help="One live system_one call via the shipped Jev client")
    live.add_argument("--snapshot", required=True)
    live.add_argument("--env", default=".env")
    _add_backend_flags(live)
    live.set_defaults(func=cmd_jev_live)

    ping = sub.add_parser("binance-ping", help="Ping Binance USD-M futures testnet")
    ping.add_argument("--env", default=".env")
    ping.set_defaults(func=cmd_binance_ping)

    summary = sub.add_parser(
        "summary",
        help="Unsigned all-symbol UM 24h ticker (public REST, no API keys)",
    )
    summary.add_argument("--once", action="store_true", help="One refresh then exit")
    summary.add_argument("--interval", type=float, default=15.0, help="Refresh seconds (default 15)")
    summary.add_argument("--max-cycles", type=int, default=None)
    summary.add_argument("--max-runtime", type=float, default=None, help="Stop after N seconds")
    summary.add_argument(
        "--full",
        action="store_true",
        help="Dump every UM symbol (default is compact: count + BTC/ETH/SOL + movers)",
    )
    summary.set_defaults(func=cmd_summary)

    def add_live_args(p: argparse.ArgumentParser, *, follow_jev: bool, fire_latest: bool) -> None:
        p.add_argument("--symbol", default="BTCUSDT")
        p.add_argument(
            "--symbols",
            default=None,
            help="Comma list, default --symbol (BTCUSDT). Optional ETHUSDT. Not the full universe.",
        )
        p.add_argument("--answers", default=None, help="Replay Jev answers JSON (skip live TypeSafe)")
        p.add_argument("--env", default=".env")
        p.add_argument("--ledger", default="data/ledger.sqlite")
        p.add_argument("--venue", choices=("paper", "testnet", "live"), default="paper")
        p.add_argument("--kill-switch", action="store_true")
        p.add_argument("--daily-pnl-pct", type=float, default=0.0)
        p.add_argument("--no-telegram", action="store_true")
        p.add_argument("--max-cycles", type=int, default=None, help="Exit after N 5m-close decisions")
        p.add_argument("--max-runtime", type=float, default=None, help="Stop after N seconds")
        p.add_argument("--kline-poll", type=float, default=5.0, help="Seconds between public kline polls")
        p.add_argument(
            "--summary-interval",
            type=float,
            default=15.0,
            help="Seconds between compact universe refreshes when --print-universe",
        )
        p.add_argument(
            "--print-universe",
            action="store_true",
            help="JSONL compact universe lines (full dump is the summary command)",
        )
        p.add_argument("--recorded-klines", default=None, help="Offline REST klines JSON")
        p.add_argument(
            "--recorded-close",
            action="append",
            default=None,
            help="Offline {symbol}@kline_5m JSON event (repeatable)",
        )
        p.add_argument("--recorded-depth", default=None, help="Offline REST depth JSON")
        p.add_argument(
            "--fire-latest",
            dest="fire_latest",
            action="store_true",
            default=fire_latest,
            help="Judge the last already-closed 5m bar immediately, then wait for the next close",
        )
        p.add_argument(
            "--wait-bar",
            dest="fire_latest",
            action="store_false",
            help="Do not judge the last bar; wait for the next 5m close",
        )
        p.add_argument(
            "--universe",
            dest="universe",
            action="store_true",
            default=True,
            help="Scan all UM symbols, Jev on the most liquid USDT perps (default)",
        )
        p.add_argument(
            "--no-universe",
            dest="universe",
            action="store_false",
            help="Only --symbol / --symbols (old single-coin mode)",
        )
        p.add_argument("--universe-size", type=int, default=DEFAULT_UNIVERSE_SIZE)
        p.add_argument("--min-quote-volume", type=float, default=DEFAULT_MIN_QUOTE_VOLUME)
        p.add_argument("--max-positions", type=int, default=5)
        p.add_argument(
            "--follow-jev",
            dest="follow_jev",
            action="store_true",
            default=follow_jev,
            help="Trade Jev buy/close without probability gates (size/stop still in code; no shorts)",
        )
        p.add_argument(
            "--strict-gates",
            dest="follow_jev",
            action="store_false",
            help="Keep should_trade_now 0.72 and other policy gates",
        )
        p.add_argument(
            "--min-should-trade",
            type=float,
            default=None,
            help="Override should_trade_now gate (default 0.72 unless --follow-jev)",
        )
        p.add_argument("--status", default=None, help="Heartbeat JSON path (default next to ledger)")
        _add_backend_flags(p)

    trade = sub.add_parser(
        "live",
        help="24/7 5m-close loop: public klines → Jev → paper/testnet",
    )
    add_live_args(trade, follow_jev=False, fire_latest=False)
    trade.set_defaults(func=cmd_live)

    desk = sub.add_parser(
        "run",
        help="Always-on bot + local trade monitor (http://127.0.0.1:8787)",
    )
    add_live_args(desk, follow_jev=True, fire_latest=True)
    desk.add_argument("--host", default="127.0.0.1")
    desk.add_argument("--port", type=int, default=8787)
    desk.add_argument("--pid", default=None, help="PID file (default next to ledger)")
    desk.set_defaults(func=cmd_run)

    mon = sub.add_parser(
        "monitor",
        help="Trade blotter in the browser (reads the same ledger as the bot)",
    )
    mon.add_argument("--ledger", default="data/ledger.sqlite")
    mon.add_argument("--status", default=None)
    mon.add_argument("--pid", default=None)
    mon.add_argument("--env", default=".env")
    mon.add_argument("--venue", choices=("paper", "testnet", "live"), default=None)
    mon.add_argument("--host", default="127.0.0.1")
    mon.add_argument("--port", type=int, default=8787)
    mon.add_argument("--json", action="store_true", help="Print blotter JSON once and exit")
    mon.set_defaults(func=cmd_monitor)

    book = sub.add_parser(
        "trades",
        help="Show fills, open position, and paper/testnet PnL from the ledger",
    )
    book.add_argument("--ledger", default="data/ledger.sqlite")
    book.add_argument("--limit", type=int, default=20)
    book.set_defaults(func=cmd_trades)

    flat = sub.add_parser(
        "flatten",
        help="Cancel open orders and MARKET-close all Binance positions (testnet/live)",
    )
    flat.add_argument("--env", default=".env")
    flat.add_argument("--ledger", default="data/ledger.sqlite")
    flat.add_argument("--venue", choices=("testnet", "live"), default="testnet")
    flat.add_argument("--pid", default=None)
    flat.add_argument(
        "--force",
        action="store_true",
        help="Flatten even if the bot PID file still looks alive",
    )
    flat.set_defaults(func=cmd_flatten)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
