"""The cache-key assumption check behind invariant 1, outside the TLA+ models.

ArtifactLifecycle.tla's two counterexamples (findings 1 and 2) and the
pruning check (finding 3) are fixed and now live under ``tests/``. The test
here passes while its bug exists (asserts the violating outcome). Once a fix
lands, invert the assertion and move the test into ``tests/``.

    uv run pytest formal/ -v
"""

from __future__ import annotations

from strata.types import CacheKey


def test_projection_fingerprint_collision() -> None:
    """Injectivity check for the row-group cache key (outside the TLA+ model).

    The fingerprint hashes ``",".join(columns)``, so projecting one column
    named ``"a,b"`` and projecting the two columns ``"a"`` and ``"b"`` give
    the same cache key. Iceberg and Parquet both allow a comma in a column
    name.
    """
    fp = CacheKey.compute_projection_fingerprint
    assert fp(["a,b"]) == fp(["a", "b"])
