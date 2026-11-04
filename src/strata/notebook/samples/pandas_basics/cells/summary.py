# @name Top product per region
# Top product by revenue in each region.
top_products = (
    sales.groupby(["region", "product"])["revenue"]
    .sum()
    .reset_index()
    .sort_values("revenue", ascending=False)
    .drop_duplicates(subset="region")
    .sort_values("region")
)

print("Top product by revenue per region:")
print(top_products.to_string(index=False))
