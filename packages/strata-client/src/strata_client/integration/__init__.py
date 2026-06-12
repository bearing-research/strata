"""Integration modules for Arrow, DataFusion, DuckDB, pandas, and Polars."""

from strata_client.integration.arrow import (
    StrataDataset,
    dataset,
)
from strata_client.integration.arrow import (
    StrataScanner as StrataArrowScanner,
)
from strata_client.integration.datafusion import (
    StrataDataFusionContext,
    register_strata_table,
)
from strata_client.integration.datafusion import (
    strata_query as datafusion_query,
)
from strata_client.integration.duckdb import (
    StrataScanner,
    StrataTableParams,
    register_strata_scan,
    strata_query,
)
from strata_client.integration.pandas import (
    StrataPandasScanner,
    scan_to_pandas,
)
from strata_client.integration.polars import (
    StrataPolarsScanner,
    scan_to_lazy,
    scan_to_polars,
)

__all__ = [
    # Arrow (foundation)
    "StrataDataset",
    "StrataArrowScanner",
    "dataset",
    # DataFusion
    "StrataDataFusionContext",
    "register_strata_table",
    "datafusion_query",
    # DuckDB
    "StrataScanner",
    "StrataTableParams",
    "register_strata_scan",
    "strata_query",
    # pandas
    "StrataPandasScanner",
    "scan_to_pandas",
    # Polars
    "StrataPolarsScanner",
    "scan_to_lazy",
    "scan_to_polars",
]
