# @name Monthly weather summary
monthly = (
    weather.assign(month=weather["DATE"].dt.to_period("M").astype(str))
    .groupby("month", as_index=False)
    .agg(
        avg_temp=("TEMP", "mean"),
        max_temp=("MAX", "max"),
        min_temp=("MIN", "min"),
        total_precip=("PRCP", "sum"),
    )
    .round(2)
)
monthly
