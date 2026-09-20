# Jev-trader

Автономный бот Binance USD-M: видит **все** фьючерсные пары, на закрытии 5m спрашивает **Jev** по самым ликвидным (плюс уже открытые позиции). Код считает размер и стоп. Jev решает `buy_long` / `sell_short` / `close` / `hold`. Вход на бирже только **BUY (лонг)**. **SELL** бывает только с `reduceOnly` — закрыть лонг. Шорт бот не открывает: если Jev сказал `sell_short` без лонга, это `hold` (`no_short`); если лонг уже есть — SELL закрывает его.

Команда, которую нужно оставить жить: `python -m jev_trader run` — бот крутится сам и открывает монитор сделок в браузере: [http://127.0.0.1:8787](http://127.0.0.1:8787).

Не инвестиционная рекомендация. По умолчанию **paper** (журнал, без биржи). `--venue testnet` — учебные ордера. Боевой Binance только с `--venue live` **и** `BINANCE_ALLOW_LIVE=I_UNDERSTAND`.

## Как устроен цикл

```
публичный рынок (без ключей)
  GET /fapi/v1/ticker/24hr     → сводка всех символов (до 15 с)
  GET /fapi/v1/klines 5m       → закрытие бара
               ↓
признаки (EMA, ATR, RSI, ADX, VWAP, HH/HL, стакан)
               →  компактный state (текст/JSON, без картинок графика)
               →  Jev: Choice + Noul + Score   (только на close 5m, не на каждый тик,
                                                не на всю вселенную символов)
               →  Policy (пороги в коде)
               →  Risk (size = risk% / ATR-стоп; вероятности Jev в формулу не входят)
               →  paper-брокер или Binance testnet
               →  SQLite ledger (+ Telegram, если включён)
```

`live` по умолчанию сканирует все UM-символы, берёт топ ликвидных USDT-perp (15, лимит 40) и на **каждом закрытии 5m** вызывает Jev по этому списку. Не 700 вызовов Jev каждые 5 минут — иначе сгорят квота и время. Открытые позиции всегда остаются в списке, даже если выпали из топа.

### Пороги входа (Policy)

Все четыре должны пройти, иначе действие — `hold`, ордера нет:

| Условие | Порог |
|---|---|
| `should_trade_now` | ≥ 0.72 |
| `false_break_risk` | ≤ 0.35 |
| `signal_strength` | не ниже `рабочий` (`нет края` < `слабый` < `рабочий` < `сильный`) |
| `trend_aligned` | ≥ 0.60 (только для входа `buy_long` / `sell_short`; для `close` не требуется) |

Jev может сказать `buy_long` и `сильный`, но если `should_trade_now = 0.36`, код стоит в стороне. Так и задумано.

Вход — LIMIT post-only (`GTX`) **BUY** по лучшему bid (иначе last close). Закрытие лонга — SELL + `reduceOnly`. SELL без `reduceOnly` брокер отвергает.

## Требования

- Python **≥ 3.12**
- ключ TypeSafe в `.env` (`typesafe_API_KEY` или `TYPESAFE_API_KEY`) — для живого Jev
- ключи Binance **testnet** — только для `--venue testnet` и `binance-ping` (ордера). Сводка и 5m-свечи ключей не требуют.

На macOS часто нет команды `python` в PATH, а системный `python3` без `typesafe_sdk`. Используйте интерпретатор из venv.

## Установка

```bash
cd /path/to/Jev-trader
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Заполните `.env` (файл в git не попадает):

```
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_FAPI_BASE=https://testnet.binancefuture.com
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_NOTIFY=0
typesafe_API_KEY=
```

`typesafe_API_KEY` программа сама мапит на `TYPESAFE_API_KEY`, который ждёт SDK.

Команды ниже — **одной строкой**. В zsh пробел после `\` превращает следующую строку (`--env`, `--ledger`) в отдельную команду: `zsh: command not found: --env`.

## Команды

Точка входа: `python -m jev_trader` (после `source .venv/bin/activate`) или `.venv/bin/python -m jev_trader`.

### Постоянный бот + монитор сделок

Это основной режим. Один процесс: на каждом закрытии 5m Jev говорит войти / держать / закрыть по ликвидным USDT-парам, код считает размер и стоп.

`--venue paper` — **локальный журнал**, стартовые 10 000 USDT, ключ Binance **не** используется.  
`--venue testnet` — виртуальный счёт Binance USD-M из `.env` (ключ + `BINANCE_FAPI_BASE=https://testnet.binancefuture.com`). Размер позиции считается от баланса биржи (~3 050 USDT, не от 10 000).

Виртуальный счёт Binance:

```bash
python -m jev_trader run --venue testnet --env .env --no-telegram
```

Только локальная тетрадка на 10 000:

```bash
python -m jev_trader run --venue paper --env .env --ledger data/ledger.sqlite --no-telegram
```

Откройте [http://127.0.0.1:8787](http://127.0.0.1:8787). Там: работает ли бот, сколько до следующей 5m, открытые позиции, исполнения, PnL, что сказал Jev и что сделал код.

`run` по умолчанию **следует действию Jev** (`buy_long` / `sell_short` / `close` / `hold`). Размер и стоп всё равно считает код, не модель. Чтобы вернуть порог `should_trade_now ≥ 0.72`, добавьте `--strict-gates`.

Остановка: Ctrl+C в том же терминале. Чтобы бот поднимался после перезагрузки Mac:

```bash
cp contrib/macos/com.jevtrader.run.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.jevtrader.run.plist
```

Только монитор, если бот уже запущен:

```bash
python -m jev_trader monitor --ledger data/ledger.sqlite
```

### 0. Сводка по всем UM-фьючерсам (без ключей)

Публичный `GET https://fapi.binance.com/fapi/v1/ticker/24hr` без `symbol` — все пары. Запрос **не** ставит `X-MBX-APIKEY` и **не** подписывается, даже если в `.env` лежат ключи. Этого достаточно, чтобы обновлять картину раз в 15 секунд.

По умолчанию в терминал идёт **компактный** JSON: `count` (сотни символов), `watch` (BTC/ETH/SOL), топ gainers/losers. Полный дамп 700+ тикеров — только с `--full`.

Один снимок:

```bash
python -m jev_trader summary --once
```

Цикл каждые 15 с (для VPS; Ctrl+C выходит без traceback):

```bash
python -m jev_trader summary --interval 15
```

Все символы в один JSON (большой):

```bash
python -m jev_trader summary --once --full
```

Пустой `count` / пустой список — ошибка, не «биржа молчит».

Ключи в дочернем процессе не нужны:

```bash
env -u BINANCE_API_KEY -u BINANCE_API_SECRET python -m jev_trader summary --once
```

### 1. Живой цикл по закрытию 5m (paper)

По умолчанию `live` смотрит **все** пары, Jev — топ-15 по обороту. `--fire-latest` сразу судит последний закрытый 5m бар (не ждать до 5 минут).

Бумажный бот 24/7 (рекомендуемый старт):

```bash
python -m jev_trader live --venue paper --env .env --ledger data/ledger.sqlite --no-telegram --fire-latest
```

Учебные ордера на Binance testnet (в `.env` должен быть `BINANCE_FAPI_BASE=https://testnet.binancefuture.com`):

```bash
python -m jev_trader live --venue testnet --env .env --ledger data/ledger.sqlite --no-telegram --fire-latest
```

Только BTC: добавьте `--no-universe --symbol BTCUSDT`.

Следовать Jev без порога 0.72 (размер всё равно считает код): `--follow-jev`.

Журнал сделок и PnL:

```bash
python -m jev_trader trades --ledger data/ledger.sqlite
```

Без `--max-cycles` процесс живёт, пока его не остановить: поллит публичные klines, на close вызывает Jev, исполняет paper/testnet, при обрыве сети переподключается.

Replay без TypeSafe (тот же пайплайн):

```bash
python -m jev_trader live --venue paper --max-cycles 1 --no-telegram --answers tests/fixtures/jev_pass.json --recorded-klines tests/fixtures/klines_5m.json
```

`--venue testnet` шлёт LIMIT+GTX только если Policy пустил; ключи должны быть testnet. Paper **никогда** не делает POST на Binance.

Опционально второй символ (всё равно не вся вселенная): `--symbols BTCUSDT,ETHUSDT`.

### 2. Один бумажный цикл из файла (живой Jev)

Фикстура `tests/fixtures/market.json` — замороженный BTCUSDT 5m, не текущий рынок.

```bash
python -m jev_trader once --snapshot tests/fixtures/market.json --venue paper --env .env --ledger data/ledger.sqlite --no-telegram
```

`--env` и `--ledger` имеют те же значения по умолчанию (`.env` и `data/ledger.sqlite`), их можно не писать.

Типичный ответ, когда Jev видит лонг, но «не сейчас»:

```json
{
  "action": "hold",
  "skip_reason": "should_trade_now",
  "intent": null,
  "execution": { "status": "skipped", "venue": "paper" },
  "judgment": {
    "action": "buy_long",
    "signal_strength": "сильный",
    "should_trade_now": 0.36,
    "model": "jev-1.13.0"
  }
}
```

Это **не ошибка**. Paper ничего не шлёт на биржу. Решение пишется в ledger.

### 3. Прогон пайплайна без живого Jev (replay)

Файл `tests/fixtures/jev_pass.json` заранее проходит все пороги (`should_trade_now: 0.84`). Ордер всё равно бумажный.

```bash
python -m jev_trader once --snapshot tests/fixtures/market.json --answers tests/fixtures/jev_pass.json --venue paper --no-telegram
```

Ожидайте `action: "buy_long"` и непустой `intent` (qty, stop, `limit_price`, `entry_type: LIMIT`).

### 4. Один цикл на Binance USD-M **testnet**

Тот же снимок и те же пороги. Если Policy не пускает — POST на биржу не уйдёт. Если пускает — LIMIT+GTX на `https://testnet.binancefuture.com`.

```bash
python -m jev_trader once --snapshot tests/fixtures/market.json --venue testnet --env .env --no-telegram
```

Ключи в `.env` должны быть **testnet**, не production.

### 5. Проверка доступа к Jev

```bash
python -m jev_trader jev-live --snapshot tests/fixtures/market.json --env .env
```

Печатает `ok`, модель и пять полей суждения. Ордеров нет.

### 6. Пинг testnet

```bash
python -m jev_trader binance-ping --env .env
```

Ожидайте `"ok": true` и HTTP 200. Если SSL падает на пустом CA store CPython.org — в проекте уже стоит `certifi`.

### 7. Kill-switch и дневной убыток

```bash
python -m jev_trader once --snapshot tests/fixtures/market.json --venue paper --kill-switch --no-telegram
python -m jev_trader once --snapshot tests/fixtures/market.json --venue paper --daily-pnl-pct -3.5 --no-telegram
```

Риск-движок не открывает новые входы (kill-switch / daily-loss flatten — см. `jev_trader/risk.py`).

### Telegram

По умолчанию `TELEGRAM_NOTIFY=0`. Чтобы слать сообщения:

1. заполните `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID`
2. поставьте `TELEGRAM_NOTIFY=1`
3. **не** передавайте `--no-telegram`

## Где происходит торговля

`summary` **не торгует** — только цены всех пар.

Сделка появляется только в `once` или `live`, и только если код после Jev **пропустил** вход:

1. закрылась 5m-свеча BTCUSDT (в `live`) или вы сами дали `--snapshot` (`once`);
2. Jev ответил `buy_long` / `sell_short` / `close`;
3. Policy прошёл (`should_trade_now ≥ 0.72` и остальные пороги);
4. Risk посчитал qty и стоп;
5. брокер исполнил: **paper** пишет бумажный fill в SQLite; **testnet** шлёт LIMIT/GTX на Binance testnet.

Пока в JSON `action=hold` и `execution.status=skipped` — ордера нет, прибыли нет. Все ваши прогоны `once` по фикстуре как раз такие.

Закрытие — не «само по стопу на бирже». Стоп считается в коде и пишется в intent. Выход — новый цикл, где Jev сказал `close` (или kill-switch / дневной убыток → flatten MARKET).

Смотреть, **что купили, когда закрыли, какой PnL**:

```bash
python -m jev_trader trades --ledger data/ledger.sqlite
```

Там: открытая позиция, `realized_pnl_usdt`, `unrealized_pnl_usdt`, список fills и последние решения (включая hold). Бумажный PnL = (цена выхода − цена входа) × qty для лонга, наоборот для шорта. Стартовый cash 10000 USDT.

Чтобы увидеть не hold, а сделку (всё ещё paper):

```bash
python -m jev_trader live --venue paper --symbol BTCUSDT --max-cycles 1 --no-telegram --answers tests/fixtures/jev_pass.json --ledger data/ledger.sqlite
python -m jev_trader trades --ledger data/ledger.sqlite
```

## Как читать JSON `once` / `live`

| Поле | Смысл |
|---|---|
| `judgment` | сырой ответ Jev (или replay из `--answers`) |
| `action` | что сделал **код** после Policy + Risk |
| `skip_reason` | какой гейт не пустил (`should_trade_now`, `false_break`, `low_strength`, `trend_aligned`, `hold`, риск) |
| `intent` | размер, стоп, LIMIT-цена; `null` если скипаем |
| `execution.status` | `skipped` / принят брокером; `venue` = `paper` или testnet |
| `state_text` | компактное состояние, которое ушло в Jev |

## 24/7 на VPS

Секреты только в `.env` (`chmod 600`), не в git и не в аргументах командной строки. Ключ Binance — **trade only, без withdraw**. Ордера — только testnet; публичные цены ключей не используют.

На машине: Python ≥ 3.12, venv, `pip install -e .`, заполненный `.env`.

Два процесса (tmux / systemd). Сводка **не** вызывает Jev:

```bash
source /path/to/Jev-trader/.venv/bin/activate
cd /path/to/Jev-trader
python -m jev_trader summary --interval 15
```

Торговый цикл (paper — безопасный старт; testnet — когда готовы к учебным ордерам):

```bash
python -m jev_trader live --venue paper --symbol BTCUSDT --env .env --ledger data/ledger.sqlite
```

```bash
python -m jev_trader live --venue testnet --symbol BTCUSDT --env .env --ledger data/ledger.sqlite
```

`--max-cycles` / `--max-runtime` на VPS не ставят. Остановка: Ctrl+C или `systemctl stop`. После разрыва публичного REST процесс сам делает backoff и продолжает. Ledger — `data/ledger.sqlite`.

Пример unit в systemd (`Restart=always`, `WorkingDirectory=` репозиторий, `EnvironmentFile=` абсолютный путь к `.env`). Не указывайте `BINANCE_FAPI_BASE=https://fapi.binance.com` для торговли: брокер это отвергнет.

## Тесты

```bash
source .venv/bin/activate
pytest
```

## Чего нет (намеренно)

- торговля на Binance production / mainnet
- вызов Jev на каждый тикер вселенной или каждые 15 с
- калибровка порогов по неделям live-данных
- user-data listenKey / Redis / Prometheus

Полная спека — в [`jev-working.md`](jev-working.md).
