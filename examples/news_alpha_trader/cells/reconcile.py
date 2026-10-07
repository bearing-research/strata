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
