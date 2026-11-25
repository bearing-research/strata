# @name The average year
cycle = (
    monthly.assign(cycle_ppm=monthly["co2_ppm"] - monthly["trend_ppm"])
    .groupby("month", as_index=False)["cycle_ppm"]
    .mean()
    .round(2)
)
cycle
