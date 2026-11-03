# @name Summarize each station
# @each path in gsod glob=725*.csv
# @write_table summary lake:gsod.stations sync on=station
import pandas as pd
import pyarrow as pa

df = pd.read_csv(path, dtype={"STATION": str})
# GSOD writes a missing reading as 9999.9.
measured = df[df["MAX"] < 999]
summary = pa.table(
    {
        "station": [df["STATION"].iloc[0]],
        "name": [str(df["NAME"].iloc[0])],
        "days": [len(df)],
        "mean_temp_f": [float(df["TEMP"].mean())],
        "hottest_f": [float(measured["MAX"].max())],
    }
)
