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
