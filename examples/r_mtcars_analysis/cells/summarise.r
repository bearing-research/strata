# @name Summarise by cylinder count
#
# Split-apply-combine in base R: group the cars by cylinder class and
# average economy, power, and weight per group. `aggregate()` is the
# base-R group-by — no dplyr needed. `cars` arrives as a data.frame
# (the Arrow IPC handoff from the prep cell), and `by_cyl` leaves as
# one too, ready for any downstream cell.

by_cyl <- aggregate(
  cbind(mpg, hp, wt) ~ cyl,
  data = cars,
  FUN = mean
)
by_cyl$n <- as.integer(table(cars$cyl))
names(by_cyl) <- c("cyl", "mean_mpg", "mean_hp", "mean_wt", "n")

print(by_cyl)
