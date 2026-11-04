# What the cache saves

Strata stores each cell's result under a hash of its source, its inputs and its
environment. If none of those changed, the cell doesn't run again. This takes about
two minutes.

## Step 1: run it

Press **Run All** at the top of the page. The cell below waits two seconds
on purpose, standing in for a slow query.
