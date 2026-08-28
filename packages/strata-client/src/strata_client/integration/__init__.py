"""Integration modules for Arrow, DataFusion, DuckDB, pandas, and Polars.

Each integration is loaded on first use rather than at import time.

This module used to import all five eagerly, which quietly made the package's
four optional extras unusable on their own: ``pip install
"strata-client[pandas]"`` followed by ``from strata_client.integration.pandas
import scan_to_pandas`` raised ``ModuleNotFoundError: No module named
'duckdb'``, because importing any submodule runs this file first and this file
imported duckdb. Only ``[all]`` worked, so the separate extras were misleading.

Names re-exported from here still resolve exactly as before (PEP 562), and the
error a caller gets for a genuinely missing dependency now names the extra they
actually need.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - resolved by type checkers only
    # ``__all__`` is built from ``_EXPORTS`` below, which linters cannot see,
    # hence the noqa on each re-export.
    from strata_client.integration.arrow import (  # noqa: F401
        StrataDataset,
        dataset,
    )
    from strata_client.integration.arrow import (  # noqa: F401
        StrataScanner as StrataArrowScanner,
    )
    from strata_client.integration.datafusion import (  # noqa: F401
        StrataDataFusionContext,
        register_strata_table,
    )
    from strata_client.integration.datafusion import (  # noqa: F401
        strata_query as datafusion_query,
    )
    from strata_client.integration.duckdb import (  # noqa: F401
        StrataScanner,
        StrataTableParams,
        register_strata_scan,
        strata_query,
    )
    from strata_client.integration.pandas import (  # noqa: F401
        StrataPandasScanner,
        scan_to_pandas,
    )
    from strata_client.integration.polars import (  # noqa: F401
        StrataPolarsScanner,
        scan_to_lazy,
        scan_to_polars,
    )

# Exported name -> (submodule, attribute in that submodule). The attribute is
# named separately because two integrations export a differently-aliased class.
_EXPORTS: dict[str, tuple[str, str]] = {
    # Arrow
    "StrataDataset": ("arrow", "StrataDataset"),
    "StrataArrowScanner": ("arrow", "StrataScanner"),
    "dataset": ("arrow", "dataset"),
    # DataFusion
    "StrataDataFusionContext": ("datafusion", "StrataDataFusionContext"),
    "register_strata_table": ("datafusion", "register_strata_table"),
    "datafusion_query": ("datafusion", "strata_query"),
    # DuckDB
    "StrataScanner": ("duckdb", "StrataScanner"),
    "StrataTableParams": ("duckdb", "StrataTableParams"),
    "register_strata_scan": ("duckdb", "register_strata_scan"),
    "strata_query": ("duckdb", "strata_query"),
    # pandas
    "StrataPandasScanner": ("pandas", "StrataPandasScanner"),
    "scan_to_pandas": ("pandas", "scan_to_pandas"),
    # Polars
    "StrataPolarsScanner": ("polars", "StrataPolarsScanner"),
    "scan_to_lazy": ("polars", "scan_to_lazy"),
    "scan_to_polars": ("polars", "scan_to_polars"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import the owning submodule on first access to one of its exports."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute)
    globals()[name] = value  # cache, so later lookups skip this path
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
