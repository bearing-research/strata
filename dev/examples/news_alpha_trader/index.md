# News Alpha Trader

**An end-to-end LLM-driven trading notebook.** Overnight news → structured sentiment signals → paper (or live) orders via Alpaca → cost-adjusted P&L, all persisted to a local DuckDB so the data accumulates across sessions.

This example is the opposite of the toy notebooks in this directory, it talks to real APIs, spends real tokens, and (if you opt in) places real orders.

## What it does

1. Pulls overnight news for a watchlist (Alpaca Market Data, Benzinga feed)
2. Extracts structured sentiment per headline via the LLM with `@output_schema` (validated JSON, auto-retried on failure)
3. Fetches latest prices and computes available capital
4. Generates a trade plan under hard risk limits (max position size, daily trade cap, ticker whitelist)
5. Submits orders, paper by default, live only behind a two-key lock
6. Reconciles fills, records realized costs (slippage, TAF, spread)
7. Reports **cost-adjusted** P&L, the only number that matters: did the strategy beat its own API and transaction bill?
8. LLM writes a daily journal entry with the day's lessons

## Required API keys

Set these in the **Runtime panel**:

| Key                   | Used for                              |
| --------------------- | ------------------------------------- |
| `ALPACA_API_KEY`      | News + prices + orders                |
| `ALPACA_API_SECRET`   | Same                                  |
| `ANTHROPIC_API_KEY` _or_ `OPENAI_API_KEY` | Signal extraction + daily note |

Alpaca free tier gives 200 req/min, 7+ years history, 15-min REST delay. Plenty for daily rebalancing.

### Secret manager

`notebook.toml` also wires a secret manager, so these keys can come from Infisical instead of being typed in:

```toml
[secret_manager]
provider = "infisical"
project_id = "7301e122-565d-491d-9082-056504381782"
```

That `project_id` is the example author's project; replace it with your own Infisical project's id. Strata logs in with credentials from the environment of the shell that starts the server, never from the notebook: export `INFISICAL_CLIENT_ID` and `INFISICAL_CLIENT_SECRET` (a Machine Identity with read access to the project, recommended) or `INFISICAL_TOKEN` (a service token), plus `INFISICAL_HOST` for a self-hosted Infisical. See [Secrets](https://bearingresearch.com/strata/latest/notebook/secrets/) for the full setup. Without those credentials the Runtime panel shows a fetch error; keys typed into the panel still work, or remove the `[secret_manager]` block to stop the fetch.

## Paper vs live

**The notebook ships in `paper` mode.** Every code path is identical between paper and live; only the Alpaca endpoint URL differs. To flip to live you must:

1. Edit `helpers.py` → set `Config.MODE = "live"`
2. Edit `helpers.py` → set `Config.I_UNDERSTAND_THIS_IS_REAL_MONEY = True`
3. Your Alpaca account must be funded

The `place_orders` cell refuses to run live without both flags set. This is intentional, two edits on purpose.

## Risk limits (hard, enforced in code)

Change in `helpers.py` → `Config` class:

- `MAX_POSITION_USD`: per-ticker notional cap (default $500)
- `MAX_TOTAL_EXPOSURE_USD`: portfolio notional cap (default $2000)
- `MAX_DAILY_TRADES`: daily trade count cap (default 10)
- `TICKER_WHITELIST`: the LLM *cannot* place orders in tickers outside this list, even if a headline mentions one

## DuckDB schema

One file: `work/trading.db` in the notebook's directory. Tables:

- `news_raw`: raw headlines (idempotent insert by `article_id`)
- `signals`: extracted sentiment per article
- `prices`: daily OHLC per ticker
- `orders`: every order attempt, expected cost, Alpaca order id
- `trades`: reconciled fills with realized cost
- `positions`: current positions (rebuilt from `trades`)
- `costs`: every dollar spent: LLM tokens, market data, commissions, TAF fees, slippage

## Running

```bash
uv sync
# Open the notebook in Strata, set API keys in the Runtime panel, run-all.
```

Re-run tomorrow, `fetch_news` only pulls new articles, `place_orders` is idempotent by `signal_id`. The DB grows over time; so does the history you can analyze.

## Disclaimers

- **This is not financial advice.** The strategy is a demo. It may lose money. Backtest thoroughly before live trading.
- LLM outputs are non-deterministic even with `temperature=0.0`. A prompt injection in a news headline could theoretically shift a signal, the ticker whitelist is the hard backstop.
- Paper fills are instantaneous and slippage-free. Live fills are not. The cost model applies estimated slippage to both modes so the numbers stay honest.
- You are responsible for understanding every line of `place_orders.py` before flipping to live.

# Read before running

This notebook is the first thing you see when you open the project. It does
nothing computationally, the real work starts in `helpers`.

## Key facts

1. **Real money mode requires two opt-ins.** This notebook can place real
   orders on your Alpaca account when both `Config.MODE == "live"` and
   `Config.I_UNDERSTAND_THIS_IS_REAL_MONEY == True`. Both live in
   `helpers.py` and default to `"paper"` / `False`. You must edit both on
   purpose to trade live.

2. **Every LLM call costs real money.** Every market-data call counts
   against your Alpaca rate limit (200/min free tier). The `costs` cell
   tracks both, check it before you leave the notebook running unattended.

3. **Paper fills aren't realistic.** Paper fills on Alpaca are instantaneous
   and slippage-free, which is not how real markets work. We apply a modeled
   slippage cost even in paper mode so the cost-adjusted P&L doesn't lie
   when you eventually flip to live.

4. **LLM outputs are non-deterministic.** A hostile news headline could in
   principle influence a signal. We enforce a ticker whitelist
   (`Config.TICKER_WHITELIST`) at the order layer, the LLM cannot place
   orders in tickers outside that list regardless of what the signal says.

5. **Nothing here is financial advice.** The strategy is a demo. Backtest on
   historical data (see `pnl_report`) before trusting it with capital.

## Helpers & Config

<sub>**kind** python</sub>

```python
# @name Helpers & Config
# Module cell: pure defs + literal constants. Downstream cells import
# Config, the DuckDB/Alpaca factories, and the cost estimators by name.
#
# Keeping ALL connection and policy knobs in one place is deliberate:
# flipping paper→live, widening the watchlist, or tightening risk is a
# single-cell edit. The DB schema also lives here so every downstream
# cell can assume the tables exist after this cell runs once.
import datetime as dt
import json
import os
from decimal import Decimal
from pathlib import Path


class Config:
    # --- Mode gate -----------------------------------------------------
    # Set MODE="live" AND I_UNDERSTAND_THIS_IS_REAL_MONEY=True to place
    # real orders. place_orders refuses to trade live without both.
    MODE = "paper"
    I_UNDERSTAND_THIS_IS_REAL_MONEY = False

    # --- Universe ------------------------------------------------------
    # Hard whitelist. place_orders drops any proposed trade whose ticker
    # is not in this list, regardless of what a signal claims.
    TICKER_WHITELIST = ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN")

    # --- Risk limits ---------------------------------------------------
    MAX_POSITION_USD = 500.0
    MAX_TOTAL_EXPOSURE_USD = 2000.0
    MAX_DAILY_TRADES = 10
    MIN_CONFIDENCE = 0.7
    MIN_ABS_SENTIMENT = 0.4

    # --- Cost model ----------------------------------------------------
    # LLM pricing is fetched from LiteLLM's public catalog at runtime;
    # see _fetch_litellm_pricing() below. Fallback rate used only when
    # the catalog can't be reached AND the model is unknown; matches
    # mid-tier Sonnet pricing so we overstate rather than log $0.
    FALLBACK_LLM_PRICE_USD_PER_M = (3.0, 15.0)

    # Alpaca equities: $0 commission. Regulatory TAF only on sells.
    TAF_USD_PER_SHARE = 0.000166
    TAF_MAX_USD_PER_TRADE = 8.30
    # Assumed half-spread / market impact applied to BOTH paper and
    # live fills so cost-adjusted P&L stays honest when you flip modes.
    SLIPPAGE_BPS = 5.0

    # --- Storage -------------------------------------------------------
    # "." is the cell's working directory, the notebook's work/.
    DB_FILENAME = "trading.db"
    LOOKBACK_DAYS = 30


def db_path() -> Path:
    """Where the DuckDB file lives (the notebook's work/)."""
    return Path(".") / Config.DB_FILENAME


def open_db():
    """Open (or create) the DuckDB file and ensure the schema exists.

    Connection is short-lived: each cell gets its own. DuckDB handles
    concurrent read-only + single-writer access fine for this scale.
    """
    import duckdb

    conn = duckdb.connect(str(db_path()))
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn) -> None:
    """Idempotent DDL. Run on every open_db() call (CREATE IF NOT EXISTS)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_raw (
            article_id BIGINT PRIMARY KEY,
            ticker VARCHAR NOT NULL,
            headline VARCHAR NOT NULL,
            summary VARCHAR,
            url VARCHAR,
            source VARCHAR,
            published_at TIMESTAMP NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS signals (
            signal_id VARCHAR PRIMARY KEY,
            article_id BIGINT NOT NULL,
            ticker VARCHAR NOT NULL,
            sentiment DOUBLE NOT NULL,
            confidence DOUBLE NOT NULL,
            theme VARCHAR,
            reasoning VARCHAR,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_tokens INTEGER,
            output_tokens INTEGER,
            model VARCHAR
        );
        CREATE TABLE IF NOT EXISTS prices (
            ticker VARCHAR NOT NULL,
            ts TIMESTAMP NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (ticker, ts)
        );
        CREATE TABLE IF NOT EXISTS orders (
            order_id VARCHAR PRIMARY KEY,
            signal_id VARCHAR,
            ticker VARCHAR NOT NULL,
            side VARCHAR NOT NULL,
            qty DOUBLE NOT NULL,
            mode VARCHAR NOT NULL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR NOT NULL,
            expected_cost_usd DOUBLE,
            alpaca_client_order_id VARCHAR
        );
        CREATE TABLE IF NOT EXISTS trades (
            trade_id VARCHAR PRIMARY KEY,
            order_id VARCHAR NOT NULL,
            ticker VARCHAR NOT NULL,
            side VARCHAR NOT NULL,
            qty DOUBLE NOT NULL,
            fill_price DOUBLE NOT NULL,
            fill_ts TIMESTAMP NOT NULL,
            realized_cost_usd DOUBLE
        );
        CREATE TABLE IF NOT EXISTS positions (
            ticker VARCHAR PRIMARY KEY,
            qty DOUBLE NOT NULL,
            avg_cost DOUBLE NOT NULL,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS costs (
            cost_id VARCHAR PRIMARY KEY,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source VARCHAR NOT NULL,
            detail_json VARCHAR,
            usd DOUBLE NOT NULL,
            signal_id VARCHAR,
            order_id VARCHAR
        );
        """
    )


def alpaca_credentials() -> tuple[str, str]:
    """Pull Alpaca key + secret from the notebook env. Raises if missing."""
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_API_SECRET not set. "
            "Add them in the Runtime panel before running data/trade cells."
        )
    return key, secret


def alpaca_trading_client():
    """Trading client pinned to the current Config.MODE.

    ``paper=True`` routes to paper-api.alpaca.markets; False routes to
    api.alpaca.markets (live). We never silently swap modes at call
    time; the caller must own the mode decision.
    """
    from alpaca.trading.client import TradingClient

    key, secret = alpaca_credentials()
    return TradingClient(key, secret, paper=(Config.MODE == "paper"))


def alpaca_data_clients() -> tuple:
    """News + historical bar clients. Data is mode-independent."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.news import NewsClient

    key, secret = alpaca_credentials()
    return StockHistoricalDataClient(key, secret), NewsClient(key, secret)


_LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)


def _fetch_litellm_pricing() -> dict:
    """Download LiteLLM's pricing catalog. Cached module-wide after first hit.

    LiteLLM maintains per-token USD rates for ~2000 models. If the fetch
    fails (offline, 403, etc.) we cache an empty dict so we don't retry
    on every cost lookup; the fallback rate takes over downstream.
    """
    global _litellm_pricing_cache  # noqa: PLW0603
    if _litellm_pricing_cache is not None:
        return _litellm_pricing_cache
    import httpx

    try:
        _litellm_pricing_cache = httpx.get(_LITELLM_PRICING_URL, timeout=5.0).json()
    except Exception as exc:
        print(f"warning: LiteLLM pricing fetch failed ({exc}); using fallback rate")
        _litellm_pricing_cache = {}
    return _litellm_pricing_cache


_litellm_pricing_cache: dict | None = None


def active_llm_model() -> str | None:
    """The model the Runtime panel last wrote to notebook.toml [ai].model.

    Returns None before the user picks one; callers then fall back to the
    conservative rate below. Notebook directory is CWD at cell execution
    time, so plain relative-path tomllib is enough.
    """
    import tomllib

    try:
        with open("notebook.toml", "rb") as f:
            return tomllib.load(f).get("ai", {}).get("model")
    except Exception:
        return None


def estimate_llm_cost(
    input_tokens: int, output_tokens: int, model: str | None = None
) -> float:
    """Dollar cost of a single LLM call.

    Pricing comes from LiteLLM's public catalog; the model defaults to
    whatever the Runtime panel selected. Unknown models fall back to
    FALLBACK_LLM_PRICE_USD_PER_M so we overstate rather than log $0.
    """
    if model is None:
        model = active_llm_model()

    entry = _fetch_litellm_pricing().get(model or "", {})
    in_per_token = entry.get("input_cost_per_token")
    out_per_token = entry.get("output_cost_per_token")

    if in_per_token is None or out_per_token is None:
        in_m, out_m = Config.FALLBACK_LLM_PRICE_USD_PER_M
        return (input_tokens / 1_000_000.0) * in_m + (output_tokens / 1_000_000.0) * out_m

    return input_tokens * in_per_token + output_tokens * out_per_token


def estimate_trade_cost(qty: float, side: str, price: float) -> dict:
    """Expected cost of placing an order.

    Three components:
    - commission (0 on Alpaca equities)
    - TAF fee (sells only, tiny)
    - slippage (Config.SLIPPAGE_BPS applied to notional, one-way)

    Applied to paper trades too. That's the point, so the backtest
    doesn't tell you a pleasant lie.
    """
    notional = abs(qty) * price
    slippage_usd = notional * (Config.SLIPPAGE_BPS / 10_000.0)
    taf_usd = 0.0
    if side.lower() == "sell":
        taf_usd = min(
            abs(qty) * Config.TAF_USD_PER_SHARE,
            Config.TAF_MAX_USD_PER_TRADE,
        )
    commission_usd = 0.0
    total = commission_usd + taf_usd + slippage_usd
    return {
        "commission_usd": commission_usd,
        "taf_usd": taf_usd,
        "slippage_usd": slippage_usd,
        "total_usd": total,
    }


def record_cost(
    conn,
    *,
    source: str,
    usd: float,
    detail: dict,
    signal_id: str | None = None,
    order_id: str | None = None,
) -> None:
    """Append a single row to the costs ledger."""
    import uuid as _uuid

    conn.execute(
        """
        INSERT INTO costs (cost_id, source, detail_json, usd, signal_id, order_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [_uuid.uuid4().hex, source, json.dumps(detail, default=str), usd, signal_id, order_id],
    )


def today_utc() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def watchlist_str() -> str:
    """Alpaca's symbols= param wants a comma-joined list."""
    return ",".join(Config.TICKER_WHITELIST)
```

## Fetch overnight news

<sub>**kind** python</sub>

```python
# @name Fetch overnight news
# Pulls recent Benzinga headlines for every whitelisted ticker and
# upserts into the ``news_raw`` table. Idempotent: re-running the
# cell only inserts articles we haven't seen before, so the DB grows
# monotonically across sessions.
import datetime as dt

import pandas as pd
from alpaca.data.requests import NewsRequest

conn = open_db()
_stock_client, news_client = alpaca_data_clients()

# Only pull since the newest article we already have for any of the
# whitelisted tickers. First run pulls Config.LOOKBACK_DAYS back.
latest_ts_row = conn.execute(
    "SELECT MAX(published_at) FROM news_raw WHERE ticker IN ({})".format(
        ",".join(f"'{t}'" for t in Config.TICKER_WHITELIST)
    )
).fetchone()
latest_ts = latest_ts_row[0] if latest_ts_row and latest_ts_row[0] else None

if latest_ts is None:
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=Config.LOOKBACK_DAYS)
else:
    # +1 second so we don't re-fetch the boundary article.
    since = (latest_ts + dt.timedelta(seconds=1)).replace(tzinfo=dt.timezone.utc)

request = NewsRequest(
    symbols=watchlist_str(),
    start=since,
    limit=50,
    include_content=False,
)
response = news_client.get_news(request)
raw_news = response.data.get("news", []) if hasattr(response, "data") else list(response)

inserted = 0
for article in raw_news:
    # alpaca-py returns a NewsSet on some SDK versions and a list on
    # others, so normalize to the underlying attrs.
    article_id = getattr(article, "id", None)
    headline = getattr(article, "headline", "") or ""
    summary = getattr(article, "summary", "") or ""
    url = getattr(article, "url", "") or ""
    source = getattr(article, "source", "") or ""
    symbols = getattr(article, "symbols", []) or []
    created_at = getattr(article, "created_at", None) or getattr(article, "published_at", None)
    if article_id is None or created_at is None:
        continue

    # An article can mention multiple tickers; explode into one row per
    # (article, ticker) so downstream joins are natural.
    for ticker in symbols:
        if ticker not in Config.TICKER_WHITELIST:
            continue
        try:
            conn.execute(
                """
                INSERT INTO news_raw
                    (article_id, ticker, headline, summary, url, source, published_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [article_id, ticker, headline, summary, url, source, created_at],
            )
            inserted += 1
        except Exception:
            # Primary key conflict: already have this article. Fine.
            pass

# Log "ingestion cost" as zero but record the row so the ledger has a
# complete audit trail of every external call.
record_cost(
    conn,
    source="alpaca_news",
    usd=0.0,
    detail={"articles_seen": len(raw_news), "inserted": inserted, "since": str(since)},
)

# What's unprocessed, used by extract_signals.
unprocessed = conn.execute(
    """
    SELECT n.article_id, n.ticker, n.headline, n.summary, n.published_at
    FROM news_raw n
    LEFT JOIN signals s ON s.article_id = n.article_id AND s.ticker = n.ticker
    WHERE s.signal_id IS NULL
    ORDER BY n.published_at DESC
    LIMIT 40
    """
).fetchdf()

conn.close()

news_ingest = {
    "new_rows": inserted,
    "unprocessed_count": len(unprocessed),
    "oldest_unprocessed": (
        str(unprocessed["published_at"].min()) if len(unprocessed) else None
    ),
}
print(
    f"news_raw: inserted {inserted} new rows; {len(unprocessed)} articles "
    f"awaiting signal extraction."
)

# Expose ``unprocessed`` so the prompt cell can reference {{ unprocessed }}.
unprocessed
```

# Turn news into signals

We've pulled raw headlines from the news API. The cell below is a
**prompt cell**: it sends each headline to an LLM with an
`@output_schema` that forces a structured JSON response (ticker,
direction, confidence). The schema-validated result becomes a
content-addressed artifact like any Python cell's output, so
identical headlines + identical prompt template = cache hit, no
extra API call.

After signal extraction, the next few cells persist them, fetch
matching price data, and run risk checks before any orders go out.

## signals_batch

<sub>**kind** prompt</sub>

*Prompt cell: response intentionally excluded from export.*

```text
# @name signals_batch
# @temperature 0.0
# @max_tokens 4096
# @system You are a quantitative-trading analyst extracting market-impact signals from news headlines. Return a row for every input article: the risk filter downstream drops weak signals, so do not pre-filter here.
# @output_schema {"type": "object", "properties": {"signals": {"type": "array", "items": {"type": "object", "properties": {"article_id": {"type": "integer"}, "ticker": {"type": "string"}, "sentiment": {"type": "number", "minimum": -1, "maximum": 1}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "theme": {"type": "string", "enum": ["earnings", "guidance", "product", "regulatory", "macro", "m_and_a", "legal", "people", "other"]}, "reasoning": {"type": "string"}}, "required": ["article_id", "ticker", "sentiment", "confidence", "theme", "reasoning"]}}}, "required": ["signals"]}
# @validate_retries 3
#
# Extracts structured sentiment for each article in the unprocessed
# batch. The schema enums are deliberately narrow: a tighter enum
# means cleaner aggregation SQL downstream. The validate-and-retry
# loop kicks in when the provider falls back to json_object mode
# (Anthropic, Gemini, Mistral) and the response doesn't quite match.
#
# Prompt-design note: earlier prompts told the model to "err toward
# low confidence on ambiguous headlines." Models read that as "omit
# the row" and returned empty arrays on unglamorous news batches.
# The current phrasing is explicit: return one row per article,
# encode uncertainty in the confidence score, don't drop articles.
# Config.MIN_CONFIDENCE + MIN_ABS_SENTIMENT at the risk layer are
# where the actual trading threshold lives.

Return exactly one signal per input article; do NOT omit rows. Encode
uncertainty in the `confidence` field. The downstream risk filter
drops low-confidence rows before any trade is placed, so your job
here is just to characterize every headline we fetched.

For each article, emit:

- `article_id` and `ticker`: copied verbatim from the input row.
- `sentiment` in [-1, 1]: -1 = clearly bearish for this ticker,
  +1 = clearly bullish, 0 = neutral / not directionally relevant.
- `confidence` in [0, 1]:
  - > 0.8 for clear-cut material news (earnings beat, explicit
    guidance, confirmed M&A, major product launch, regulatory
    action with a named target).
  - 0.5–0.8 for solid analyst coverage, sector news with clear
    ticker impact, leadership moves with stated effect.
  - 0.2–0.5 for rumors, side mentions, ambiguous macro.
  - < 0.2 for headlines that aren't really about this ticker
    (cross-references, generic market commentary). Still return
    the row; use 0 sentiment and explain in `reasoning`.
- `theme`: pick the single best enum value.
- `reasoning`: one short sentence explaining the score.

Do NOT infer signals for tickers other than the one on each row.

Articles:

{{ unprocessed }}
```

## Persist signals

<sub>**kind** python</sub>

```python
# @name Persist signals
# Writes the freshly-extracted batch to the ``signals`` table and logs
# the LLM call to the ``costs`` ledger so cost-adjusted P&L has
# something to subtract later.
import hashlib

conn = open_db()
batch = signals_batch or {}
rows = batch.get("signals", []) if isinstance(batch, dict) else []

# Only persist rows whose ticker is in the whitelist. The schema enum
# doesn't constrain ticker strings (the LLM could return anything),
# so this is the last layer of defense before the DB has a row that
# place_orders would try to trade.
allowed = set(Config.TICKER_WHITELIST)
persisted = 0
for row in rows:
    ticker = row.get("ticker")
    article_id = row.get("article_id")
    if ticker not in allowed or article_id is None:
        continue
    # Deterministic signal_id so re-running the cell is idempotent.
    seed = f"{article_id}:{ticker}".encode()
    signal_id = hashlib.sha256(seed).hexdigest()[:16]
    try:
        conn.execute(
            """
            INSERT INTO signals
                (signal_id, article_id, ticker, sentiment, confidence,
                 theme, reasoning, input_tokens, output_tokens, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                signal_id,
                article_id,
                ticker,
                float(row.get("sentiment", 0.0)),
                float(row.get("confidence", 0.0)),
                str(row.get("theme", "other")),
                str(row.get("reasoning", "")),
                None,  # populated below if transform_spec carries it
                None,
                None,
            ],
        )
        persisted += 1
    except Exception:
        # Primary-key conflict on re-run: signal already persisted.
        pass

# LLM cost accounting. The prompt cell doesn't expose tokens to
# downstream cells directly; we pull them off the signals_batch
# artifact's transform_spec at reconcile time, so here we just log a
# placeholder row and let the reconciliation cell fill in the real
# dollar figure when it inspects the artifact's metadata. For now,
# estimate from the template+response size so the ledger always has
# an entry even if reconciliation is skipped.
approx_in_tokens = int(len(str(unprocessed)) / 4)
approx_out_tokens = int(
    len(str(batch)) / 4 if isinstance(batch, dict) else 0
)
approx_cost = estimate_llm_cost(approx_in_tokens, approx_out_tokens)
record_cost(
    conn,
    source="llm_signals",
    usd=approx_cost,
    detail={
        "rows_in": len(rows),
        "persisted": persisted,
        "approx_input_tokens": approx_in_tokens,
        "approx_output_tokens": approx_out_tokens,
        "note": "estimated from template size; exact token counts in artifact",
    },
)

# Fresh signals ready for risk_check.
fresh_signals = conn.execute(
    """
    SELECT s.signal_id, s.ticker, s.sentiment, s.confidence, s.theme,
           s.extracted_at, n.headline
    FROM signals s
    JOIN news_raw n USING (article_id, ticker)
    WHERE s.signal_id NOT IN (SELECT COALESCE(signal_id, '') FROM orders)
      AND s.confidence >= ?
      AND ABS(s.sentiment) >= ?
    ORDER BY s.extracted_at DESC
    """,
    [Config.MIN_CONFIDENCE, Config.MIN_ABS_SENTIMENT],
).fetchdf()

conn.close()
print(f"persisted {persisted} new signals; {len(fresh_signals)} meet risk thresholds")
fresh_signals
```

## Fetch latest prices

<sub>**kind** python</sub>

```python
# @name Fetch latest prices
# Pulls daily bars for the watchlist up to "now" and upserts into
# ``prices``. We need a recent close to size positions; the 15-min
# REST delay is fine here, since daily rebalancing doesn't need sub-minute
# precision.
import datetime as dt

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

conn = open_db()
stock_client, _news = alpaca_data_clients()

start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=Config.LOOKBACK_DAYS)
request = StockBarsRequest(
    symbol_or_symbols=list(Config.TICKER_WHITELIST),
    timeframe=TimeFrame.Day,
    start=start,
)
bar_response = stock_client.get_stock_bars(request)
# alpaca-py returns a BarSet; .df is a MultiIndex DataFrame keyed by
# (symbol, timestamp). Normalize to rows.
frame = bar_response.df.reset_index()

inserted = 0
for row in frame.itertuples(index=False):
    try:
        conn.execute(
            """
            INSERT INTO prices (ticker, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            [
                row.symbol,
                row.timestamp,
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                int(row.volume),
            ],
        )
        inserted += 1
    except Exception:
        pass

record_cost(
    conn,
    source="alpaca_data",
    usd=0.0,  # free tier
    detail={"bars_fetched": len(frame), "inserted": inserted},
)

# Latest close per ticker, which risk_check uses for sizing.
latest_prices = conn.execute(
    """
    SELECT ticker, close, ts AS as_of
    FROM prices
    WHERE (ticker, ts) IN (
        SELECT ticker, MAX(ts) FROM prices GROUP BY ticker
    )
    ORDER BY ticker
    """
).fetchdf()

conn.close()
print(f"prices: upserted {inserted} bars; latest for {len(latest_prices)} tickers.")
latest_prices
```

## Risk check & trade plan

<sub>**kind** python</sub>

```python
# @name Risk check & trade plan
# Translates fresh signals into a concrete list of orders after
# applying all hard risk constraints. Nothing here talks to Alpaca;
# the output is a DataFrame that place_orders either submits or drops
# on dry-run.
import pandas as pd

conn = open_db()

# --- Current state -----------------------------------------------------
current_positions = conn.execute(
    "SELECT ticker, qty, avg_cost FROM positions WHERE qty != 0"
).fetchdf()
open_exposure = 0.0
if len(current_positions):
    # Use last known close for exposure math.
    current_positions = current_positions.merge(
        latest_prices[["ticker", "close"]], on="ticker", how="left"
    )
    current_positions["notional"] = (
        current_positions["qty"] * current_positions["close"]
    ).abs()
    open_exposure = float(current_positions["notional"].sum())

orders_today = conn.execute(
    """
    SELECT COUNT(*) FROM orders
    WHERE CAST(submitted_at AS DATE) = CURRENT_DATE
    """
).fetchone()[0]
trades_budget = max(0, Config.MAX_DAILY_TRADES - int(orders_today))

# --- Signals → trade plan ---------------------------------------------
#
# Simple rule: one entry per fresh signal, sized proportionally to
# signal strength (sentiment × confidence), capped at
# MAX_POSITION_USD. No pyramiding on existing positions: if we
# already hold the ticker, skip.
trade_plan_rows = []
held = set(current_positions["ticker"]) if len(current_positions) else set()
price_lookup = dict(zip(latest_prices["ticker"], latest_prices["close"], strict=False))
headroom_usd = max(0.0, Config.MAX_TOTAL_EXPOSURE_USD - open_exposure)

for signal in fresh_signals.itertuples(index=False):
    if len(trade_plan_rows) >= trades_budget:
        break
    if signal.ticker in held:
        continue
    price = price_lookup.get(signal.ticker)
    if not price or price <= 0:
        continue
    strength = float(signal.sentiment) * float(signal.confidence)
    if abs(strength) < (Config.MIN_CONFIDENCE * Config.MIN_ABS_SENTIMENT):
        continue
    # Long on positive, short on negative. Alpaca Basic supports
    # shorting on margin; set strength positive-only to disable shorts.
    side = "buy" if strength > 0 else "sell"
    # Size = |strength| × position cap, further clamped by headroom.
    target_usd = min(abs(strength) * Config.MAX_POSITION_USD, headroom_usd)
    if target_usd < price:
        # Can't afford even one share within limits.
        continue
    qty = int(target_usd // price)
    if qty <= 0:
        continue
    expected = estimate_trade_cost(qty, side, price)
    trade_plan_rows.append(
        {
            "signal_id": signal.signal_id,
            "ticker": signal.ticker,
            "side": side,
            "qty": qty,
            "price_hint": price,
            "strength": strength,
            "theme": signal.theme,
            "expected_cost_usd": expected["total_usd"],
        }
    )
    headroom_usd -= qty * price
    if headroom_usd <= 0:
        break

# Empty lists produce a DataFrame with zero columns, which trips up
# the downstream filters in place_orders. Pin the schema.
_TRADE_PLAN_COLUMNS = [
    "signal_id",
    "ticker",
    "side",
    "qty",
    "price_hint",
    "strength",
    "theme",
    "expected_cost_usd",
]
trade_plan = pd.DataFrame(trade_plan_rows, columns=_TRADE_PLAN_COLUMNS)

conn.close()

print(
    f"risk_check: {len(fresh_signals)} fresh signals → {len(trade_plan)} proposed "
    f"orders. Open exposure ${open_exposure:.2f}, trades budget left {trades_budget}."
)
trade_plan
```

# Place orders, reconcile, report

Signal extraction and risk checks land cached artifacts upstream.
Below is the execution side: send the trade plan to the broker,
reconcile what filled, attribute trading costs, and write the
end-of-day P&L into a structured report.

Editing any analysis cell above invalidates the downstream
execution cells through provenance, so a re-run never sends stale
orders. The very last cell is a prompt cell that summarizes the
day's session for human review.

## Place orders

<sub>**kind** python</sub>

```python
# @name Place orders
# Takes the trade_plan from risk_check and either DRY-RUNs it or
# submits each row to Alpaca. Re-running is idempotent: if a
# signal_id is already in ``orders``, we skip it.
#
# The live-mode gate is intentionally verbose. Flipping from paper
# to live takes two separate edits in helpers.py, and this cell enforces
# the second one at runtime.
import uuid

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

conn = open_db()

# Resolve mode once so logs are consistent even if Config changes
# mid-run.
mode = Config.MODE

# Hard gate: refuses to trade live without the second key.
if mode == "live" and not Config.I_UNDERSTAND_THIS_IS_REAL_MONEY:
    raise RuntimeError(
        "Config.MODE is 'live' but Config.I_UNDERSTAND_THIS_IS_REAL_MONEY is False. "
        "Set both in helpers.py on purpose before placing real orders."
    )

if mode not in ("paper", "live"):
    raise ValueError(f"Config.MODE must be 'paper' or 'live', got {mode!r}.")

# Deduplicate against anything already submitted today.
already_submitted = set(
    row[0]
    for row in conn.execute(
        "SELECT signal_id FROM orders WHERE signal_id IS NOT NULL"
    ).fetchall()
)
if len(trade_plan) == 0 or "signal_id" not in trade_plan.columns:
    to_submit = trade_plan.iloc[0:0]  # empty but with compatible shape
    skipped_dupe = 0
else:
    to_submit = trade_plan[~trade_plan["signal_id"].isin(already_submitted)]
    skipped_dupe = len(trade_plan) - len(to_submit)

submitted_rows = []
if len(to_submit):
    client = alpaca_trading_client()
    for row in to_submit.itertuples(index=False):
        client_order_id = f"strata-{row.signal_id}-{uuid.uuid4().hex[:6]}"
        request = MarketOrderRequest(
            symbol=row.ticker,
            qty=int(row.qty),
            side=OrderSide.BUY if row.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        try:
            alpaca_order = client.submit_order(order_data=request)
        except Exception as exc:
            conn.execute(
                """
                INSERT INTO orders
                    (order_id, signal_id, ticker, side, qty, mode,
                     status, expected_cost_usd, alpaca_client_order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    uuid.uuid4().hex,
                    row.signal_id,
                    row.ticker,
                    row.side,
                    float(row.qty),
                    mode,
                    f"submit_failed: {type(exc).__name__}",
                    float(row.expected_cost_usd),
                    client_order_id,
                ],
            )
            continue

        order_id = str(getattr(alpaca_order, "id", uuid.uuid4().hex))
        status = str(getattr(alpaca_order, "status", "submitted"))
        conn.execute(
            """
            INSERT INTO orders
                (order_id, signal_id, ticker, side, qty, mode,
                 status, expected_cost_usd, alpaca_client_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                order_id,
                row.signal_id,
                row.ticker,
                row.side,
                float(row.qty),
                mode,
                status,
                float(row.expected_cost_usd),
                client_order_id,
            ],
        )
        # Expected-cost is booked to ``costs`` now; reconcile will
        # add any delta (realized_cost − expected) after fills land.
        record_cost(
            conn,
            source="trade_expected",
            usd=float(row.expected_cost_usd),
            detail={
                "ticker": row.ticker,
                "side": row.side,
                "qty": float(row.qty),
                "price_hint": float(row.price_hint),
                "mode": mode,
            },
            signal_id=row.signal_id,
            order_id=order_id,
        )
        submitted_rows.append(
            {
                "order_id": order_id,
                "ticker": row.ticker,
                "side": row.side,
                "qty": int(row.qty),
                "status": status,
                "expected_cost_usd": float(row.expected_cost_usd),
            }
        )

conn.close()
import pandas as pd

# Pin columns so daily_note and reconcile see a consistent schema
# even on a no-op run.
_SUBMITTED_COLUMNS = ["order_id", "ticker", "side", "qty", "status", "expected_cost_usd"]
submitted = pd.DataFrame(submitted_rows, columns=_SUBMITTED_COLUMNS)
print(
    f"place_orders ({mode}): {len(submitted)} submitted, "
    f"{skipped_dupe} skipped (already in orders)."
)
submitted
```

## Reconcile fills

<sub>**kind** python</sub>

```python
# @name Reconcile fills
# Pulls the authoritative state from Alpaca for every order we've
# submitted in the last 24h, records fills to ``trades``, rebuilds
# ``positions``, and books the slippage delta (realized - expected)
# to the cost ledger.
#
# Depends on ``submitted`` to force DAG ordering after ``place_orders``.
# Even when there are zero new submissions we still want to
# reconcile any open orders from prior runs.
import datetime as dt
import uuid

from alpaca.trading.requests import GetOrdersRequest

# Reference submitted so the DAG runs us after place_orders. The
# variable itself is unused: reconcile pulls authoritative state
# from Alpaca regardless of what we submitted this run.
_upstream_gate = submitted  # noqa: F841

conn = open_db()
client = alpaca_trading_client()

# Pull every order we've submitted recently. Alpaca returns status in
# one canonical string per order; fill_price / fill_qty land on the
# order object when filled.
request = GetOrdersRequest(
    status="all",
    after=(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=3)),
    limit=100,
)
alpaca_orders = client.get_orders(filter=request) or []

recorded_fills = 0
for ao in alpaca_orders:
    order_id = str(getattr(ao, "id", ""))
    if not order_id:
        continue
    status = str(getattr(ao, "status", ""))
    filled_qty = float(getattr(ao, "filled_qty", 0) or 0)
    filled_price = getattr(ao, "filled_avg_price", None)
    filled_at = getattr(ao, "filled_at", None)
    ticker = str(getattr(ao, "symbol", ""))
    side = str(getattr(ao, "side", "")).lower()

    # Update the orders table's status in case the submit happened in
    # a previous run (e.g. DAY order filled overnight).
    conn.execute(
        "UPDATE orders SET status = ? WHERE order_id = ?",
        [status, order_id],
    )

    if status not in ("filled", "partially_filled"):
        continue
    if filled_qty <= 0 or filled_price is None or filled_at is None:
        continue

    # Idempotent: one trade row per (order_id, fill_ts).
    existing = conn.execute(
        "SELECT 1 FROM trades WHERE order_id = ? AND fill_ts = ?",
        [order_id, filled_at],
    ).fetchone()
    if existing:
        continue

    fill_price = float(filled_price)
    realized = estimate_trade_cost(filled_qty, side, fill_price)
    trade_id = uuid.uuid4().hex
    conn.execute(
        """
        INSERT INTO trades
            (trade_id, order_id, ticker, side, qty, fill_price, fill_ts, realized_cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            trade_id,
            order_id,
            ticker,
            side,
            filled_qty,
            fill_price,
            filled_at,
            realized["total_usd"],
        ],
    )

    # Slippage delta vs the estimate booked by place_orders.
    expected_row = conn.execute(
        "SELECT expected_cost_usd FROM orders WHERE order_id = ?", [order_id]
    ).fetchone()
    expected = float(expected_row[0]) if expected_row and expected_row[0] else 0.0
    delta = realized["total_usd"] - expected
    if abs(delta) > 0.001:
        record_cost(
            conn,
            source="trade_slippage_delta",
            usd=delta,
            detail={
                "ticker": ticker,
                "side": side,
                "qty": filled_qty,
                "fill_price": fill_price,
                "expected_cost_usd": expected,
                "realized_cost_usd": realized["total_usd"],
            },
            order_id=order_id,
        )

    recorded_fills += 1

# Rebuild ``positions`` from scratch using signed qty.
conn.execute("DELETE FROM positions")
conn.execute(
    """
    INSERT INTO positions (ticker, qty, avg_cost, last_update)
    SELECT
        ticker,
        SUM(CASE WHEN side = 'buy' THEN qty ELSE -qty END) AS qty,
        CASE
            WHEN SUM(CASE WHEN side = 'buy' THEN qty ELSE 0 END) = 0 THEN 0
            ELSE SUM(CASE WHEN side = 'buy' THEN qty * fill_price ELSE 0 END)
                 / SUM(CASE WHEN side = 'buy' THEN qty ELSE 0 END)
        END AS avg_cost,
        CURRENT_TIMESTAMP
    FROM trades
    GROUP BY ticker
    HAVING SUM(CASE WHEN side = 'buy' THEN qty ELSE -qty END) != 0
    """
)

positions_snapshot = conn.execute(
    "SELECT ticker, qty, avg_cost FROM positions ORDER BY ticker"
).fetchdf()

conn.close()
print(
    f"reconcile: recorded {recorded_fills} new fills; "
    f"{len(positions_snapshot)} open positions."
)
positions_snapshot
```

## Cost ledger

<sub>**kind** python</sub>

```python
# @name Cost ledger
# Aggregates the ``costs`` table so the next cell can subtract the
# right number from P&L. Also produces a human-readable per-source
# breakdown and a running daily total.
#
# Depends on ``positions_snapshot`` so the DAG orders us after
# reconcile; otherwise costs posted during reconciliation
# (slippage deltas) would miss this aggregation.
_upstream_gate = positions_snapshot  # noqa: F841

conn = open_db()

cost_by_source = conn.execute(
    """
    SELECT source, ROUND(SUM(usd), 4) AS usd, COUNT(*) AS calls
    FROM costs
    GROUP BY source
    ORDER BY usd DESC
    """
).fetchdf()

cost_by_day = conn.execute(
    """
    SELECT CAST(ts AS DATE) AS day, ROUND(SUM(usd), 4) AS usd
    FROM costs
    GROUP BY day
    ORDER BY day DESC
    LIMIT 14
    """
).fetchdf()

total_cost_usd = float(
    conn.execute("SELECT COALESCE(SUM(usd), 0) FROM costs").fetchone()[0]
)

conn.close()

print("=== Cost breakdown by source ===")
print(cost_by_source.to_string(index=False))
print()
print("=== Daily spend (last 14d) ===")
print(cost_by_day.to_string(index=False))
print()
print(f"Total spend to date: ${total_cost_usd:.4f}")

cost_summary = {
    "total_usd": round(total_cost_usd, 4),
    "by_source": cost_by_source.to_dict(orient="records"),
    "by_day": cost_by_day.to_dict(orient="records"),
}
cost_summary
```

## P&L report

<sub>**kind** python</sub>

```python
# @name P&L report
# The money number: realized P&L minus everything in ``costs``.
# Re-running the notebook over time builds up history here.
#
# References cost_summary + positions_snapshot to lock DAG order
# after costs + reconcile.
_upstream_gate_costs = cost_summary  # noqa: F841
_upstream_gate_positions = positions_snapshot  # noqa: F841

import pandas as pd

conn = open_db()

# Realized P&L is SUM(sell proceeds) - SUM(buy costs) per ticker,
# only valid for tickers where quantity has round-tripped back to
# zero. Mark unrealized P&L separately against the latest close.
realized_by_ticker = conn.execute(
    """
    SELECT
        ticker,
        SUM(CASE WHEN side = 'sell' THEN qty * fill_price ELSE 0 END)
        - SUM(CASE WHEN side = 'buy'  THEN qty * fill_price ELSE 0 END) AS realized_pnl_usd,
        SUM(qty) AS turnover
    FROM trades
    GROUP BY ticker
    ORDER BY realized_pnl_usd DESC
    """
).fetchdf()

unrealized_by_ticker = conn.execute(
    """
    WITH latest AS (
        SELECT ticker, close
        FROM prices
        WHERE (ticker, ts) IN (
            SELECT ticker, MAX(ts) FROM prices GROUP BY ticker
        )
    )
    SELECT p.ticker,
           p.qty,
           p.avg_cost,
           l.close AS mark_price,
           ROUND(p.qty * (l.close - p.avg_cost), 2) AS unrealized_pnl_usd
    FROM positions p
    LEFT JOIN latest l USING (ticker)
    WHERE p.qty != 0
    ORDER BY p.ticker
    """
).fetchdf()

total_realized = float(realized_by_ticker["realized_pnl_usd"].sum()) if len(realized_by_ticker) else 0.0
total_unrealized = (
    float(unrealized_by_ticker["unrealized_pnl_usd"].sum()) if len(unrealized_by_ticker) else 0.0
)
total_costs = float(conn.execute("SELECT COALESCE(SUM(usd), 0) FROM costs").fetchone()[0])

# The headline metric: does the strategy pay for itself?
cost_adjusted_pnl = total_realized + total_unrealized - total_costs

conn.close()

print("=== Realized P&L by ticker ===")
print(realized_by_ticker.to_string(index=False) if len(realized_by_ticker) else "(no closed trades yet)")
print()
print("=== Unrealized P&L (mark-to-last-close) ===")
print(
    unrealized_by_ticker.to_string(index=False) if len(unrealized_by_ticker) else "(no open positions)"
)
print()
print("=== Headline ===")
print(f"  Realized P&L:        ${total_realized:,.2f}")
print(f"  Unrealized P&L:      ${total_unrealized:,.2f}")
print(f"  Total costs:         ${total_costs:,.4f}")
print(f"  Cost-adjusted P&L:   ${cost_adjusted_pnl:,.2f}")

pnl_summary = {
    "realized_usd": round(total_realized, 2),
    "unrealized_usd": round(total_unrealized, 2),
    "costs_usd": round(total_costs, 4),
    "cost_adjusted_pnl_usd": round(cost_adjusted_pnl, 2),
    "by_ticker_realized": realized_by_ticker.to_dict(orient="records"),
    "by_ticker_unrealized": unrealized_by_ticker.to_dict(orient="records"),
}
pnl_summary
```

## daily_note

<sub>**kind** prompt</sub>

*Prompt cell: response intentionally excluded from export.*

```text
# @name daily_note
# @temperature 0.2
# @max_tokens 800
# @system You are a trading journal assistant. Your job is to produce a concise, honest end-of-day reflection on a small paper-trading strategy. Do not sugarcoat: if the strategy is losing money or being eaten by costs, say so directly.
# @output_schema {"type": "object", "properties": {"one_liner": {"type": "string"}, "biggest_winner": {"type": "string"}, "biggest_loser": {"type": "string"}, "cost_health": {"type": "string", "enum": ["green", "yellow", "red"]}, "cost_comment": {"type": "string"}, "lessons": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5}, "tomorrow_watch": {"type": "array", "items": {"type": "string"}}}, "required": ["one_liner", "biggest_winner", "biggest_loser", "cost_health", "cost_comment", "lessons", "tomorrow_watch"]}
# @validate_retries 2
#
# Last cell in the notebook. Produces a structured end-of-day journal
# entry from the P&L summary and cost ledger, the kind of thing
# you'd want to read first the next morning.

Trading day recap. Produce an honest, structured summary.

## P&L

{{ pnl_summary }}

## Costs

{{ cost_summary }}

## Today's orders

{{ submitted }}

## Rules for the journal

- `one_liner`: one sentence summarizing the day. Lead with the
  cost-adjusted P&L sign.
- `biggest_winner` / `biggest_loser`: name the ticker and amount;
  use "none" if no closed trades yet.
- `cost_health`: "green" when costs are < 10% of gross P&L,
  "yellow" at 10–40%, "red" at >40% or when P&L is negative.
- `cost_comment`: one sentence explaining the color.
- `lessons`: 1-5 short imperatives ("cut losers faster", "avoid
  low-confidence signals", etc.). If the sample size is too small to
  conclude anything, say so in one of the lessons.
- `tomorrow_watch`: specific things to look at tomorrow. Can be
  empty if there's nothing actionable yet.

Be terse. No filler.
```
