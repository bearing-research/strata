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
