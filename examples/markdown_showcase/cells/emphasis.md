## Inline emphasis

This is **bold** and so is __this__. This is *italic* and so is _this_.
This is ***both*** at once. This is ~~struck through~~ via GFM.

Code spans use backticks: `df.groupby("ticker").agg("sum")`. Inline
code should keep `_underscores_` and `*asterisks*` literal, they're
data, not formatting.

Edge cases the old renderer was bad at:

- Mid-word italic shouldn't fire: `snake_case_name` stays as one token.
- Math-style underscores: `x_1, x_2, x_3` should also stay literal.
- Asterisks inside code spans: `a * b * c` keeps the stars.
- A pair of backticks containing a backtick: `` `like_this` ``.
