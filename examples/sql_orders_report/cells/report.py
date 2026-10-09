# @name report
"""Combine the two SQL results into a markdown report.

The SQL cells return Arrow Tables, which reach this cell as
``pyarrow.Table``; ``.to_pandas()`` gives the familiar ``.iterrows()``.
"""

lines = [
    "# Orders report",
    "",
    f"## Top {len(top_orders)} orders above ${min_amount}",
    "",
    "| customer | sku | category | amount | ordered |",
    "|---|---|---|---:|---|",
]
for _, row in top_orders.to_pandas().iterrows():
    lines.append(
        f"| {row['customer']} | {row['sku']} | {row['category']} | "
        f"${row['amount']:.2f} | {row['ordered_at']} |"
    )

lines += [
    "",
    "## Revenue by product category",
    "",
    "| category | SKUs | orders | revenue |",
    "|---|---:|---:|---:|",
]
for _, row in category_summary.to_pandas().iterrows():
    revenue = row["total_revenue"]
    revenue_str = f"${revenue:.2f}" if revenue is not None else "n/a"
    lines.append(
        f"| {row['category']} | {int(row['sku_count'])} | "
        f"{int(row['order_count'])} | {revenue_str} |"
    )

report = "\n".join(lines)
print(report)
report
