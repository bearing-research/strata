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
