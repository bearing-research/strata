# @name Prepare the mtcars data
#
# A pure-R notebook — every cell is R. Variables flow cell-to-cell
# through the same content-addressed artifact store the Python cells
# use: a data.frame crosses as Arrow IPC, so the next R cell receives
# it as a data.frame with no glue code.
#
# `mtcars` ships with R (1974 Motor Trend, 32 cars). Tidy it into the
# frame the rest of the notebook builds on: keep the model name as a
# real column, make the cylinder count a factor (so the model and the
# plots treat it as categorical), and keep the columns we care about.

cars <- data.frame(
  model = rownames(mtcars),
  mpg = mtcars$mpg,
  wt = mtcars$wt,
  hp = mtcars$hp,
  cyl = factor(mtcars$cyl),
  row.names = NULL
)

cat(sprintf(
  "Prepared %d cars across %d cylinder classes (%s)\n",
  nrow(cars), nlevels(cars$cyl), paste(levels(cars$cyl), collapse = ", ")
))
