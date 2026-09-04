## Unordered

- alpha
- beta
- gamma

## Ordered

1. first
2. second
3. third

## Nested

- top level
  - nested level
    - deeply nested
  - back to nested
- another top level
  1. nested ordered one
  2. nested ordered two

## Mixed (ordered + unordered, with content)

1. **Step one**: set up the environment.
   - Install dependencies via `uv sync`.
   - Verify with `uv run pytest tests/notebook/`.
2. **Step two**: run the notebook.
   - Open it via the Strata UI.
   - Click `Run All` to execute every Python cell in order.
3. **Step three**: read the output.
