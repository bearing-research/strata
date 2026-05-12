# @name Merge with region metadata
region_info = pd.DataFrame(
    {
        "region": ["North", "South", "East", "West"],
        "manager": ["Alice", "Bob", "Carol", "Dave"],
        "target_revenue": [50000, 45000, 55000, 40000],
    }
)

region_summary = (
    sales.groupby("region")["revenue"].sum().reset_index().merge(region_info, on="region")
)

region_summary["pct_of_target"] = (
    100 * region_summary["revenue"] / region_summary["target_revenue"]
).round(1)

print(region_summary)
