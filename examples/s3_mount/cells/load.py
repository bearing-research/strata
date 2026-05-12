# @name Load JFK weather from S3
import pandas as pd

weather = pd.read_csv(jfk_weather / "72503014732.csv")
weather["DATE"] = pd.to_datetime(weather["DATE"])
weather = weather[["DATE", "NAME", "TEMP", "MAX", "MIN", "PRCP"]]
weather
