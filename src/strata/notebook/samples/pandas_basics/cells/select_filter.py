# @name Filter high-value orders
# Filter to high-value orders (units > 20 and price > 30).
high_value = sales[(sales["units"] > 20) & (sales["price"] > 30)].copy()

print(
    f"High-value orders: {len(high_value)} / {len(sales)} ({100 * len(high_value) / len(sales):.1f}%)"
)
high_value.head()
