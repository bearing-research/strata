# @name Summarize by region
by_region = sales.groupby("region", as_index=False)["revenue"].sum()
by_region
