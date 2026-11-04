## Step 3: edit one cell

In the cell above, change `.sum()` to `.mean()`. It reads `stale`, and so does the chart
below, which uses its result. The load cell stays `cached`. Press **Run All**: only those
two run.

Now change it back to `.sum()` and press **Run All** once more. Both read `cached` at once,
because Strata still has that result.
