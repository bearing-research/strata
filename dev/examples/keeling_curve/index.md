# The Keeling curve, with a receipt

![The Keeling curve, with a receipt: what a run of the notebook shows](https://bearingresearch.com/strata/dev/assets/examples/keeling_curve-light.webp#only-light)
![The Keeling curve, with a receipt: what a run of the notebook shows](https://bearingresearch.com/strata/dev/assets/examples/keeling_curve-dark.webp#only-dark)

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

# The Keeling curve, with a receipt

In 1958 Charles David Keeling began measuring carbon dioxide in the air at Mauna Loa,
Hawaii. The record has run every month since, and it is the clearest picture there is
of what we have added to the atmosphere.

This notebook downloads that record from NOAA, splits out the yearly breathing of the
northern forests, measures how fast the level is rising, and draws the curve. Then it
publishes the figure with a receipt: a link where anyone can see the chart, the code
behind it, and the exact bytes NOAA served.

Press **Run All** at the top of the page. The first run downloads about 40 KB.

## Load the Mauna Loa record

<sub>**kind** python</sub>

```python
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
```

## The yearly cycle

Through each northern summer the forests draw carbon dioxide down, and the level
bottoms out in early autumn; through the winter it comes back, and peaks in May. NOAA
publishes the record with that cycle removed as well, and the difference between the
two is the cycle itself. Averaged by month, it is the shape the curve repeats every year.

## The average year

<sub>**kind** python</sub>

```python
# @name The average year
cycle = (
    monthly.assign(cycle_ppm=monthly["co2_ppm"] - monthly["trend_ppm"])
    .groupby("month", as_index=False)["cycle_ppm"]
    .mean()
    .round(2)
)
cycle
```

## How fast it is rising

The yearly mean, compared with the year before, gives the growth rate in parts per
million per year. Averaged by decade, it shows whether the rise is steady or speeding up.
Only complete years count, so a year joins once NOAA publishes its December, and the
`years` column says how many each decade has so far.

## Growth per decade

<sub>**kind** python</sub>

```python
# @name Growth per decade
full_years = monthly.groupby("year").filter(lambda year: len(year) == 12)
yearly = full_years.groupby("year")["co2_ppm"].mean()
growth = (
    yearly.diff()
    .dropna()
    .groupby(lambda year: f"{year // 10 * 10}s")
    .agg(ppm_per_year="mean", years="count")
    .round({"ppm_per_year": 2})
    .rename_axis("decade")
    .reset_index()
)
growth
```

## The curve

The thin line is each month's mean; the thick one has the yearly cycle removed. On the
right, the average year: the drop through each northern summer, lowest in
September and October.

Change something here: the title, a color, the size. Only this cell runs again. The
record is not downloaded again, because nothing the earlier cells read has changed.

## The Keeling curve

<sub>**kind** python</sub>

```python
# @name The Keeling curve
import matplotlib.pyplot as plt

plt.style.use("strata")


def draw(record, average_year):
    fig, (curve, year) = plt.subplots(
        1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [4, 1]}
    )

    curve.plot(record["date"], record["co2_ppm"], linewidth=1, label="monthly mean")
    curve.plot(record["date"], record["trend_ppm"], linewidth=2.5, label="yearly cycle removed")
    latest = record.iloc[-1]
    curve.annotate(
        f"{latest['co2_ppm']:.1f} ppm\n{latest['date']:%B %Y}",
        (latest["date"], latest["co2_ppm"]),
        xytext=(-15, -100),
        textcoords="offset points",
        ha="right",
        arrowprops={"arrowstyle": "-", "color": "0.6", "linewidth": 0.8},
    )
    curve.set_title("CO₂ at Mauna Loa, parts per million")
    curve.legend(loc="upper left")

    year.bar(average_year["month"], average_year["cycle_ppm"], color="C2")
    year.set_xticks([1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"])
    year.set_title("The average year")

    fig.tight_layout()
    return fig


draw(monthly, cycle)
```

## The receipt

Every result above is already stored with what produced it: the source of each cell,
the environment it ran in, and, for the first cell, the URL it read with the digest of
the bytes NOAA served and when they were downloaded. Publishing the figure makes that
readable by anyone with the link.

The figure's id is `nb_<notebook id>_cell_plot_var___display__0`, where the notebook id
is the `notebook_id` on the first line of this notebook's `notebook.toml` (a copy of a
sample gets its own). From the notebook's folder, which the home page lists under its
title, with the server running:

```bash
strata artifact publish nb_<notebook id>_cell_plot_var___display__0 \
  --title "The Keeling curve" --artifact-dir .strata/artifacts
```

The page it prints shows the chart, this notebook's code for it and for every step
behind it, and NOAA's file under **External inputs**.

NOAA rewrites the file about once a month, usually with a new month in it. When it does,
the first cell turns stale on its own, because the bytes behind its URL changed, and so
does everything after it. Run All again and the curve takes in the new month; the figure
you published keeps the data it was drawn from.

*Data: NOAA Global Monitoring Laboratory and Scripps Institution of Oceanography,
monthly mean CO₂ at Mauna Loa (gml.noaa.gov/ccgg/trends). From December 2022 to
July 2023, after the eruption of Mauna Loa, measurements came from Maunakea, about 21
miles north.*
