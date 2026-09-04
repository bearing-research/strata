# Limitations

Slicing has a hard rule: the synthetic module is built from **one cell's
source only**: it can't reach across cells to find names. Three failure
modes follow from that:

1. **Closures over runtime values.** A helper that references a value
   computed at runtime in the same cell can't be exported, because
   that value won't exist when the synthetic module re-executes.
2. **Cross-cell imports.** `import math` in cell A doesn't make `math`
   visible inside a helper exported by cell B, each cell that hosts
   library code has to carry its own imports.
3. **Cross-cell helpers.** A helper can't call another helper that
   lives in a different cell. Move them into the same cell, or
   duplicate the dependency.

The next cell triggers the first failure mode on purpose so you can see
the diagnostic, it should error with a message naming the unresolved
variable.
