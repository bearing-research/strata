# @name Load the data
import time

import pandas as pd

time.sleep(2)

sales = pd.DataFrame(
    {
        "region": ["North", "North", "South", "South", "East", "East", "West", "West"],
        "quarter": ["Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2"],
        "revenue": [120, 135, 98, 110, 143, 151, 87, 95],
    }
)
sales
