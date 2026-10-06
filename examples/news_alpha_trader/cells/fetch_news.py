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
