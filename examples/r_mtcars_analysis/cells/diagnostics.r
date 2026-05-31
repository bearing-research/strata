# @name Residual diagnostics (base graphics)
#
# `plot()` on an lm object draws R's canonical 2x2 diagnostic panel
# (residuals vs fitted, Q-Q, scale-location, residuals vs leverage).
# It's a base-graphics plot, and it's captured to PNG the same way the
# ggplot is — no extra code, no device juggling. `par(mfrow)` tiles the
# four panels onto one image.
#
# `model` arrives by reading the RDS artifact the fit cell stored: an
# R-only object (an lm, S3 class and all) flowing from one R cell to
# another with full fidelity — something the tabular Arrow tier can't
# carry. This is the R-to-R counterpart of the cross-language Arrow
# handoff.

op <- par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(model)
par(op)
