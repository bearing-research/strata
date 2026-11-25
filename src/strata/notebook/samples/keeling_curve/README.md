# The Keeling curve, with a receipt

Atmospheric CO₂ at Mauna Loa since 1958, from NOAA's monthly record: the curve, the
yearly cycle split out, the growth rate by decade, and a published figure whose page
shows the chart, the code behind it and the exact bytes NOAA served.

Open a copy with `strata tutorial keeling_curve`, or from **Open a sample** on the home
page. The first run downloads about 40 KB from NOAA.

## What it shows

- **`# @fetch`.** The first cell declares the URL it reads. The bytes' digest is part of
  the cell's key, so when NOAA adds a month the cell goes stale by itself, and each
  stored result records the URL, the digest and when it was downloaded.
- **Caching.** Edit the plot and only the plot runs again; the record is not downloaded
  again.
- **Publishing.** `strata artifact publish` turns the figure into a link. The page lists
  the code of every step behind the figure and NOAA's file under External inputs.
- **`plt.style.use("strata")`.** The chart reads on the light and the dark theme.

## Cells

1. `load.py`: downloads the record and keeps the monthly mean and NOAA's
   seasonally adjusted series
2. `cycle.py`: the average year, the monthly mean minus the adjusted series, by month
3. `growth.py`: the growth rate in ppm per year, averaged by decade, over complete years
4. `plot.py`: the curve and the average year, side by side

The markdown cells between them explain each step and how to publish the figure.

## Data

NOAA Global Monitoring Laboratory and Scripps Institution of Oceanography, monthly mean
CO₂ at Mauna Loa: <https://gml.noaa.gov/ccgg/trends/data.html>. From December 2022 to
July 2023, after the eruption of Mauna Loa, measurements came from Maunakea, about 21
miles north.
