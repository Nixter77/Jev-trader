# Jev × Binance: рабочий план автономного торгового бота

> Документ-спецификация. Не инвестиционная рекомендация.  
> Модель: TypeSafe AI **Jev** (System One, text-only, early access, сентябрь 2026).  
> Биржа: Binance Spot / USD-M Futures. Старт только с Testnet / paper.

---

## 0. Коротко

Jev **не смотрит на график глазами** и **не пишет текст**.  
На вход — сжатое состояние рынка (свечи, индикаторы, стакан, позиция, новости).  
На выход — типизированные решения с вероятностями за ~70–500 мс:

| Примитив | Что возвращает | Пример в боте |
|---|---|---|
| `Choice` | один вариант из закрытого списка + вероятности | `buy_long / sell_short / close / hold` |
| `Noul` | P(да) ∈ [0, 1] | «ложный пробой?», «торговать сейчас?» |
| `Score` | положение на заданной шкале | сила сигнала: нет края → сильный |

**Правило архитектуры:** код считает признаки, размер, стоп и лимиты.  
Jev только **судит сигнал**. Сайзинг и риск модели не отдавать.

```mermaid
flowchart LR
  A[График / стакан / новости] --> B[Feature engine]
  B --> C[State builder<br/>компактный текст/JSON]
  C --> D[Jev API<br/>Choice + Noul + Score]
  D --> E[Policy<br/>пороги уверенности]
  E --> F[Risk engine]
  F --> G[Binance execution]
  G --> H[Ledger + kill-switch]
  H -.-> B
```

---

## 1. Зачем Jev, а не LLM

```mermaid
flowchart TB
  subgraph LLM["Обычный LLM"]
    L1[Токен за токеном] --> L2[Свободный текст]
    L2 --> L3[Парсинг JSON]
    L3 --> L4[Ошибки типа / галлюцинации]
    L4 --> L5[Секунды–минуты, дорого]
  end

  subgraph JEV["Jev System One"]
    J1[Один параллельный проход] --> J2[Типизированный ответ]
    J2 --> J3[Вероятности калиброваны]
    J3 --> J4[Схема фиксирована заранее]
    J4 --> J5[70–500 мс, ~$0.042 / 1M input]
  end
```

| Параметр | Jev 1.13 | Frontier LLM |
|---|---|---|
| Выход | Choice / Score / Noul | свободный текст |
| Latency | 70–500 мс | секунды–минуты |
| Цена входа | $0.042 / 1M tok | существенно выше |
| Выходные токены | не тарифицируются | тарифицируются |
| Картинки графика | нет (text-only) | зависит от модели |
| Гарантия схемы | да | нет, нужен парсер |

Пинить версию в проде: `jev-1.13.0`, не `jev-latest` — иначе поедут пороги.

---

## 2. Целевая архитектура бота

```mermaid
flowchart TB
  subgraph IN["Входы"]
    W1[Binance WS<br/>kline / aggTrade / depth / markPrice]
    W2[REST snapshot<br/>funding / OI / баланс]
    W3[Новостной буфер<br/>3–5 заголовков / час]
  end

  subgraph CORE["Ядро"]
    FE[Feature engine<br/>EMA RSI ATR ADX VWAP структура]
    SB[State builder]
    JEV[Jev client<br/>typesafe-sdk]
    POL[Policy / confidence gates]
    RISK[Risk engine<br/>size stop daily-loss cooldown]
    EXEC[Execution adapter]
  end

  subgraph OUT["Выходы"]
    ORD[Binance orders<br/>limit / post-only / reduceOnly]
    LED[Ledger SQLite/Redis]
    MON[Метрики + Telegram]
    KILL[Kill-switch]
  end

  W1 --> FE
  W2 --> FE
  W3 --> SB
  FE --> SB
  SB --> JEV
  JEV --> POL
  POL --> RISK
  RISK --> EXEC
  EXEC --> ORD
  ORD --> LED
  LED --> MON
  RISK --> KILL
  KILL --> EXEC
  LED -.-> SB
```

### Цикл принятия решения

Не каждый тик. На CEX разумный ритм — **закрытие свечи** или **событие** (пробой свинга, всплеск объёма, скачок funding).

```mermaid
sequenceDiagram
  participant WS as Binance WS
  participant FE as Feature engine
  participant SB as State builder
  participant J as Jev API
  participant P as Policy + Risk
  participant X as Execution
  participant L as Ledger

  WS->>FE: kline close / depth
  FE->>SB: индикаторы + структура
  SB->>J: state + questions
  J-->>P: action, noul, score, probs
  alt пороги пройдены и риск ок
    P->>X: limit / reduceOnly
    X->>L: fill / reject
  else hold или блок риска
    P->>L: skip + причина
  end
```

---

## 3. Как «анализ графика» без зрения

Jev не ест PNG TradingView. График сериализуется в короткий state.

```mermaid
flowchart LR
  C[Свечи OHLCV] --> S[Сжатие<br/>% change, не сырые цены]
  I[Индикаторы] --> S
  ST[Структура HH/HL/LH/LL] --> S
  B[Стакан L2] --> S
  POS[Позиция бота] --> S
  N[Новости] --> S
  S --> J[Jev]
```

### Минимальный снапшот (BTCUSDT 5m)

```text
symbol=BTCUSDT tf=5m ts=2026-09-18T14:35Z
close=108420 ret_1=0.12% ret_5=0.41% atr14=0.38%
ema20>ema50>ema200 adx=27 rsi=61 vwap_dist=+0.22%
structure=HH_HL last_swing_low=-0.9%
book_imb=+0.18 spread=1.2bps
pos=FLAT cash_usdt=10000
news: "ETF inflow +$240m; no FOMC today"
```

| Блок state | Содержание | Зачем |
|---|---|---|
| Цена | close, ret_1 / ret_5 / ret_12 | импульс без сырого тика |
| Вола | ATR%, диапазон бара / ATR | стоп считает код, Jev видит режим |
| Тренд | EMA20/50/200, ADX | фильтр направления |
| Осциллятор | RSI, dist-to-VWAP | перегрев / возврат |
| Структура | HH/HL или LH/LL, дистанция до свинга | контекст пробоя |
| Стакан | imbalance top-5, spread bps | микроструктура входа |
| Позиция | side, size, entry, uPnL%, время в сделке | close ≠ новый вход |
| Макро-микро | funding, ΔOI 1h, корреляция с BTC | режим фьючерсов |
| Текст | 3–5 заголовков | Jev сильна на тексте |

Лимит: порядка 32k токенов на state + вопросы. Не класть 500 сырых свечей.

---

## 4. Схема вопросов к Jev (ядро v1)

Один вызов `system_one` — несколько вопросов параллельно.

```mermaid
flowchart TB
  S[State рынка] --> Q
  subgraph Q["Один запрос Jev"]
    A["Choice action<br/>buy_long / sell_short / close / hold"]
    B["Noul trend_aligned"]
    C["Noul false_break_risk"]
    D["Score signal_strength<br/>нет края → слабый → рабочий → сильный"]
    E["Noul should_trade_now"]
  end
  Q --> P{Policy}
  P -->|все гейты зелёные| T[Исполнение]
  P -->|иначе| H[HOLD + лог причины]
```

### Формулировки

**1. Choice `action`**  
Инструкция: выбрать действие на горизонте 6–12 баров рабочего ТФ с учётом тренда старшего ТФ, импульса, структуры и стакана.  
Критерии:

- `buy_long` — преимущество у роста, вход в лонг оправдан;
- `sell_short` — преимущество у снижения, вход в шорт оправдан;
- `close` — открытую позицию лучше закрыть;
- `hold` — края нет, ждать.

**2. Noul `trend_aligned`**  
«Движение согласовано с режимом вышестоящего ТФ?»

**3. Noul `false_break_risk`**  
«Высокий риск ложного пробоя текущего уровня / свинга?»

**4. Score `signal_strength`**  
Шкала: `нет края` / `слабый` / `рабочий` / `сильный`.

**5. Noul `should_trade_now`**  
«Сейчас стоит открывать, а не ждать следующий бар?»

Не просить у Jev: целевую цену, размер позиции, «почему рынок упадёт».

---

## 5. Policy: пороги (код, не модель)

Стартовые значения — **гипотеза**. Калибровать на сохранённых ответах Jev, не угадывать.

```mermaid
flowchart TD
  A[Ответ Jev] --> B{action ∈ buy_long, sell_short, close?}
  B -->|нет| H[HOLD]
  B -->|да| C{should_trade_now ≥ 0.72?}
  C -->|нет| H
  C -->|да| D{false_break_risk ≤ 0.35?}
  D -->|нет| H
  D -->|да| E{signal_strength ≥ рабочий?}
  E -->|нет| H
  E -->|да| F{trend_aligned ≥ 0.60<br/>для нового входа?}
  F -->|нет| H
  F -->|да| G[Передать в Risk engine]
```

| Гейт | Старт | Смысл |
|---|---|---|
| `should_trade_now` | ≥ 0.72 | отсечь «почти» |
| `false_break_risk` | ≤ 0.35 | не ловить шипы |
| `signal_strength` | ≥ «рабочий» | не торговать шум |
| `trend_aligned` | ≥ 0.60 на вход | не контртренд v1 |
| `close` | можно ослабить `trend_aligned` | выход важнее входа |

Все отказы писать в лог: `skip_reason=false_break|low_strength|…`.

---

## 6. Risk engine и исполнение Binance

```mermaid
flowchart TB
  SIG[Сигнал Policy] --> R0{Kill-switch активен?}
  R0 -->|да| FLAT[Flatten + пауза]
  R0 -->|нет| R1{Дневной убыток ≥ лимита?}
  R1 -->|да| FLAT
  R1 -->|нет| R2{Есть место по notional / корреляции?}
  R2 -->|нет| SKIP[Skip]
  R2 -->|да| R3[Размер = риск% / ATR-стоп]
  R3 --> R4[Лимитный / post-only вход]
  R4 --> R5[Стоп 1.2–1.8×ATR + reduceOnly]
  R5 --> R6{Fill?}
  R6 -->|да| LED[Ledger + трейлинг по правилам кода]
  R6 -->|нет / timeout| CXL[Cancel + лог]
```

### Жёсткие лимиты (Jev не имеет права ломать)

| Правило | Старт v1 |
|---|---|
| Риск на сделку | 0.3–0.7% депозита |
| Стоп | 1.2–1.8 × ATR рабочего ТФ |
| Плечо | 2–5× isolated |
| Одновременно позиций | 1–3 |
| Корреляция | лимит совместного BTC-beta |
| Дневной стоп счёта | −2…−3% → flatten до завтра |
| Вход | limit / post-only |
| Выход по аварии | market + reduceOnly |
| Cooldown после стопа | N баров |
| API-ключ | trade only, **без withdraw** |
| События | блок входа при высоком news_shock |

### Исполнение

- Futures: `POST /fapi/v1/order`, `reduceOnly` на закрытии.
- User Data Stream — fill / partial / cancel.
- Идемпотентность: стабильный `clientOrderId`.
- В симе сразу закладывать комиссию 0.02–0.04% и проскальзывание.
- Сверять локальный ledger с позицией биржи на каждом цикле.

---

## 7. Стек

```mermaid
flowchart LR
  PY[Python 3.12] --> SDK[typesafe-sdk]
  PY --> BN[ccxt или binance-futures-connector]
  PY --> FE2[pandas / ta / свой расчёт]
  PY --> DB[(SQLite + Redis)]
  PY --> TG[Telegram alerts]
  PY --> PROM[latency / errors metrics]
```

| Слой | Выбор |
|---|---|
| Язык | Python 3.12 |
| Jev | `pip install typesafe-sdk`, env `TYPESAFE_API_KEY` |
| Биржа | Binance Testnet → live sub-account |
| Данные | WS kline + depth, REST для funding/баланса |
| Хранение | SQLite сделки, Redis горячий state |
| Деплой | маленький VPS ближе к региону API Binance |
| Секреты | только env, ключ без withdraw |

Ориентиры по чужому коду (не копировать в live blindly):

- `jarrodwatts/jev-trader` — частый цикл, Choice buy/sell;
- `tyleree/jevbot` — Jev как ядро, риск в коде;
- `Spykoninho/trading-bot-jev` — Binance + Jev по новостям;
- `zadescoxp/Jev-Trades` — paper + индикаторы.

---

## 8. Этапы внедрения

```mermaid
gantt
  title Дорожная карта v1
  dateFormat  YYYY-MM-DD
  axisFormat  %d.%m

  section Доступы
  TypeSafe key + Binance testnet     :a0, 2026-09-19, 2d

  section Каркас
  WS, фичи, paper broker без Jev     :a1, after a0, 5d

  section Jev
  State + questions + лог ответов    :a2, after a1, 7d

  section Калибровка
  Сэмпл вызовов + replay порогов     :a3, after a2, 14d

  section Paper
  24/7 BTC+ETH, дашборд, алерты      :a4, after a3, 21d

  section Микро-live
  Капитал, который можно потерять    :a5, after a4, 7d
```

### Этап 0 — доступы (1–2 дня)

- Ключ TypeSafe, `TYPESAFE_API_KEY`.
- Binance Testnet + отдельный sub-account.
- Репозиторий, секреты вне git.

### Этап 1 — каркас без Jev (3–5 дней)

- Стрим kline + depth.
- Индикаторы + state builder.
- Бумажный брокер и ledger.
- Dummy-policy RSI/EMA, чтобы ордерный пайплайн жил.

### Этап 2 — Jev как судья (3–7 дней)

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")

response = client.system_one(
    state=market_state_text,
    questions={
        "action": Choice(
            instructions="Действие на горизонте 6–12 баров рабочего ТФ.",
            criteria={
                "buy_long": "Преимущество у роста, вход в лонг.",
                "sell_short": "Преимущество у снижения, вход в шорт.",
                "close": "Открытую позицию лучше закрыть.",
                "hold": "Края нет, ждать.",
            },
        ),
        "trend_aligned": Noul(
            instructions="Движение согласовано с режимом старшего ТФ."
        ),
        "false_break_risk": Noul(
            instructions="Высокий риск ложного пробоя."
        ),
        "signal_strength": Score(
            instructions="Сила края.",
            criteria=["нет края", "слабый", "рабочий", "сильный"],
        ),
        "should_trade_now": Noul(
            instructions="Стоит открывать сейчас, а не ждать следующий бар."
        ),
    },
)
```

Логировать request, response, mid-цену через 6 и 12 баров.

### Этап 3 — бэктест / walk-forward (1–2 недели)

Jev платная, бесплатного historical replay нет.

```mermaid
flowchart LR
  H[История свечей] --> S[Сэмпл точек решения]
  S --> J[Разовые вызовы Jev]
  J --> STORE[(Сохранённые ответы)]
  STORE --> R[Replay порогов без API]
  R --> M[PF / DD / %hold / калибровка]
```

1. Базовая история без Jev.
2. На сэмпле точек — вызвать Jev, сохранить ответы.
3. Крутить пороги на сохранённых ответах.
4. Hold-out отрезок не трогать при подборе порогов.

Метрики: profit factor, max DD, доля `hold`, калибровка вероятностей, cost per decision.

Ожидание по ранним публичным прогонам: Jev часто много `hold`, край без фильтров слабый. Не закладывать «+50% в месяц».

### Этап 4 — paper 24/7 (2–4 недели)

- Пары: BTCUSDT, ETHUSDT.
- Дашборд: action, probs, позиция, PnL, latency, ошибки.
- Telegram: дневной стоп, рассинхрон позиции, p95 latency > 800 мс.

### Этап 5 — микро-live

- Сумма, которую можно потерять целиком.
- Те же пороги, что на бумаге.
- Неделя наблюдения → только потом масштаб.

---

## 9. Стоимость и частота вызовов

| Величина | Ориентир |
|---|---|
| Цена входа | $0.042 / 1M токенов |
| Выход | $0 |
| Latency | 70–500 мс |
| Rate limit (публично) | порядка 1200 req/min |
| Размер снапшота | ~400–1500 токенов |
| Ритм v1 | 2 пары × 5m ≈ 576 вызовов/сутки |

Узкое место — качество вопросов и калибровка, не счёт TypeSafe.  
Не вызывать Jev на каждый тик стакана.

```mermaid
flowchart LR
  A[Тик стакана] -->|нет| X[Игнорировать]
  B[Закрытие 5m] -->|да| J[Jev]
  C[Событие: пробой / объём / funding] -->|да| J
```

---

## 10. Усиления после v1

```mermaid
flowchart TB
  V1[v1: 5m + 5 вопросов] --> V2[Мульти-ТФ 5m+1h+4h в одном state]
  V1 --> V3[Новости: Choice bullish / hawkish / noise]
  V1 --> V4[Режим рынка: trend / range / shock]
  V4 --> V5[Разные пороги на режим]
  V1 --> V6[Ансамбль: Jev AND правило EMA200]
```

- Мульти-ТФ: один state, не три дорогих вызова.
- Новости судит Jev, торговать/нет решает код.
- Режим рынка отдельным `Choice` → разные гейты.
- Не плодить модели: один Jev, разные схемы вопросов.

---

## 11. Чего не делать

```mermaid
flowchart TB
  BAD1[Кормить скриншот графика] --> FAIL[Не работает: text-only]
  BAD2[Просить свободный комментарий] --> FAIL2[Не тот интерфейс]
  BAD3[size = вероятность × депозит] --> FAIL3[Слив без ATR-стопа]
  BAD4[Оптимизация порогов на том же отрезке] --> FAIL4[Переобучение]
  BAD5[Jev ставит стоп и плечо] --> FAIL5[Нет контура риска]
  BAD6[Withdraw на торговом ключе] --> FAIL6[Операционный риск]
```

- Не публиковать бенчмарки Jev, если это запрещает customer agreement.
- Не считать сам факт «новой модели» альфой.  
  Альфа = state + вопросы + риск + исполнение.

---

## 12. Карта ответственности

```mermaid
flowchart LR
  subgraph CODE["Код"]
    C1[Фичи и сжатие графика]
    C2[Размер позиции]
    C3[Стоп / тейк / трейлинг]
    C4[Лимиты и kill-switch]
    C5[Ордера и сверка]
  end

  subgraph JEV2["Jev"]
    J1[Выбор действия]
    J2[P ложного пробоя]
    J3[Сила сигнала]
    J4[P торговать сейчас]
  end
```

Если контур слева слабый, скорость Jev не спасёт.

---

## 13. Чеклист запуска paper

- [ ] `TYPESAFE_API_KEY` и модель `jev-1.13.0`
- [ ] Binance Testnet, ключ без withdraw
- [ ] State builder даёт стабильный сжатый текст
- [ ] Пять вопросов v1 логируются целиком
- [ ] Policy гейты в коде, не «как скажет модель»
- [ ] ATR-стоп и дневной лимит убытка
- [ ] Ledger сверяется с биржей
- [ ] Алерты latency / flatten / рассинхрон
- [ ] Есть hold-out для порогов
- [ ] Понимание: большинство таких ботов на live не повторяют бумагу

---

## 14. Словарь

| Термин | Значение здесь |
|---|---|
| System One | модель быстрых типизированных решений, не чат |
| Choice / Noul / Score | три примитива ответа Jev |
| State | сжатое описание рынка + позиции |
| Policy | пороги на вероятностях |
| Risk engine | размер, стоп, лимиты, kill-switch |
| Replay | пересчёт правил на сохранённых ответах без новых вызовов API |
| Post-only | лимитный вход maker, без снятия спреда |

---

*Версия документа: 2026-09-19. Источник: план внедрения Jev на Binance (testnet-first).*
