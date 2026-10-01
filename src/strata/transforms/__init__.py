"""Transforms: base class, built-ins (scan@v1, duckdb_sql@v1), registry and build runner."""

from strata.transforms.base import (
    Transform,
    _run_transform,
    get_transform,
    list_transforms,
    register_transform,
    run_transform,  # Deprecated, use _run_transform internally
)
from strata.transforms.build_store import (
    BuildState,
    BuildStore,
    get_build_store,
)
from strata.transforms.duckdb_sql import (
    DuckDBSQLParams,
    DuckDBSQLTransform,
    build_duckdb_sql_transform,
)
from strata.transforms.registry import (
    TransformDefinition,
    TransformRegistry,
    get_transform_registry,
)
from strata.transforms.runner import (
    BuildRunner,
    RunnerConfig,
    get_build_runner,
    set_build_runner,
)
from strata.transforms.scan import (
    ScanParams,
    ScanTransform,
    build_scan_transform,
)

__all__ = [
    # Core transform abstraction
    "Transform",
    "_run_transform",  # Internal: for server/embedded executor use
    "get_transform",
    "list_transforms",
    "register_transform",
    "run_transform",  # Deprecated: kept for backward compatibility
    # Built-in transforms
    "DuckDBSQLParams",
    "DuckDBSQLTransform",
    "ScanParams",
    "ScanTransform",
    "build_duckdb_sql_transform",
    "build_scan_transform",
    # Server-mode infrastructure
    "BuildRunner",
    "BuildState",
    "BuildStore",
    "RunnerConfig",
    "TransformDefinition",
    "TransformRegistry",
    "get_build_runner",
    "get_build_store",
    "get_transform_registry",
    "set_build_runner",
]
