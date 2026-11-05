# @name Add computed columns
orders = sales.copy()
orders["revenue"] = orders["units"] * orders["price"]
orders["month"] = orders["date"].dt.to_period("M").astype(str)

print(f"Total revenue: ${orders['revenue'].sum():,.2f}")
orders[["date", "region", "product", "units", "price", "revenue", "month"]].head()
