# @name Use the mixed-cell helpers
#
# This cell consumes both the library code (``clamp``, ``CLAMP_MIN``)
# *and* a runtime variable (``raw_max``) from the producing cell;
# the slicer routes each through the right path automatically.

clamped_examples = {
    "raw_max_clamped": clamp(raw_max),  # raw_max is runtime → artifact path
    "clamp_at_floor": clamp(-99),
    "clamp_at_ceil": clamp(999),
    "clamp_min_constant": CLAMP_MIN,  # CLAMP_MIN is literal → module path
}
