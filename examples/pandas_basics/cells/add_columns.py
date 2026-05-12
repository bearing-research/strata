# @name Add computed columns
sales["revenue"] = sales["units"] * sales["price"]
sales["month"] = sales["date"].dt.to_period("M").astype(str)

print(f"Total revenue: ${sales['revenue'].sum():,.2f}")
sales[["date", "region", "product", "units", "price", "revenue", "month"]].head()
