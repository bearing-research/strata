## The receipt

Every result above is already stored with what produced it: the source of each cell,
the environment it ran in, and, for the first cell, the URL it read with the digest of
the bytes NOAA served and when they were downloaded. Publishing the figure makes that
readable by anyone with the link. From this notebook's folder, with the server running:

```bash
strata artifact publish nb_keeling-curve-001_cell_plot_var___display__0 \
  --title "The Keeling curve" --artifact-dir .strata/artifacts
```

The page it prints shows the chart, this notebook's code for it and for every step
behind it, and NOAA's file under **External inputs**.

NOAA adds a month to the record early each month. When it does, the first cell turns
stale on its own, because the bytes behind its URL changed, and so does everything
after it. Run All again and the curve is a month longer; the figure you published keeps
the data it was drawn from.

*Data: NOAA Global Monitoring Laboratory and Scripps Institution of Oceanography,
monthly mean CO₂ at Mauna Loa (gml.noaa.gov/ccgg/trends). From December 2022 to
July 2023, after the eruption of Mauna Loa, measurements came from Maunakea, about 21
miles north.*
