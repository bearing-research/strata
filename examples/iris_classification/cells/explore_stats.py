# @name Explore per-feature stats
# Summary statistics grouped by species.
stats = df.groupby("species").agg(["mean", "std"]).round(2)
print(stats)
