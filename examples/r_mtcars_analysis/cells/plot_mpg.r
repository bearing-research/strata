# @name Plot mpg vs weight (ggplot2)
#
# The headline 0.2.0 R feature: plots render inline as PNG, exactly
# like a Python matplotlib figure. A bare trailing ggplot object
# auto-prints (REPL-style), so no explicit `print()` is needed — the
# plot is the cell's last expression and it just shows up.
#
# ggplot2 isn't part of the harness baseline (arrow + jsonlite); it
# comes from this notebook's renv.lock, restored automatically when you
# open the notebook — or one click via the Environment panel's
# "Initialize renv". If it's somehow missing you'll see a structured
# "install ggplot2" hint, not a crash.

library(ggplot2)

ggplot(cars, aes(x = wt, y = mpg, colour = cyl)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.7) +
  labs(
    title = "Fuel economy falls with weight",
    subtitle = "mtcars, coloured by cylinder count",
    x = "Weight (1000 lbs)",
    y = "Miles per gallon",
    colour = "Cylinders"
  ) +
  theme_minimal(base_size = 13)
