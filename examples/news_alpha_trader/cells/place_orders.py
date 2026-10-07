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
