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

One file: `trading.db` next to the notebook. Tables:

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

## ⚠️ Disclaimers

- **This is not financial advice.** The strategy is a demo. It may lose money. Backtest thoroughly before live trading.
- LLM outputs are non-deterministic even with `temperature=0.0`. A prompt injection in a news headline could theoretically shift a signal, the ticker whitelist is the hard backstop.
- Paper fills are instantaneous and slippage-free. Live fills are not. The cost model applies estimated slippage to both modes so the numbers stay honest.
- You are responsible for understanding every line of `place_orders.py` before flipping to live.
