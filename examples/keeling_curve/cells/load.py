# @name Load the Mauna Loa record
# @fetch co2 https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv
import pandas as pd

raw = pd.read_csv(co2, comment="#")
monthly = pd.DataFrame(
    {
        "date": pd.to_datetime(raw[["year", "month"]].assign(day=15)),
        "year": raw["year"],
        "month": raw["month"],
        "co2_ppm": raw["average"],
        "trend_ppm": raw["deseasonalized"],
    }
)
monthly
