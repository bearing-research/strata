# @name Growth per decade
full_years = monthly.groupby("year").filter(lambda year: len(year) == 12)
yearly = full_years.groupby("year")["co2_ppm"].mean()
growth = (
    yearly.diff()
    .dropna()
    .groupby(lambda year: f"{year // 10 * 10}s")
    .mean()
    .round(2)
    .rename("ppm_per_year")
    .rename_axis("decade")
    .reset_index()
)
growth
