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
