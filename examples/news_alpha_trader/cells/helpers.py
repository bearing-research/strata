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
    # "." resolves to the notebook directory at runtime.
    DB_FILENAME = "trading.db"
    LOOKBACK_DAYS = 30


def db_path() -> Path:
    """Where the DuckDB file lives (notebook directory)."""
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
