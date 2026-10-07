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
