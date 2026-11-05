# @name Revenue by region and product
region_product = (
    orders.groupby(["region", "product"])["revenue"]
    .agg(["sum", "mean", "count"])
    .round(2)
    .sort_values("sum", ascending=False)
)

print(region_product)
