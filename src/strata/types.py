"""Core types for Strata."""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

# Filter types live in the dependency-free ``strata.filters`` module so the
# client can use them without importing this (pydantic-laden) module. Re-export
# them here for backward compatibility — ``from strata.types import Filter`` etc.
# keep working.
from strata.filters import (  # noqa: F401
    Filter,
    FilterOp,
    FilterValue,
    SupportsOrdering,
    compute_filter_fingerprint,
)

if TYPE_CHECKING:
    import pyarrow as pa


# ---------------------------------------------------------------------------
# Authentication / Authorization Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """Authenticated identity from trusted proxy.

    Represents the user or service making a request, as identified by
    a trusted upstream proxy. Strata does not perform authentication
    itself - it trusts identity headers injected by the proxy.

    Attributes:
        id: Stable user/service identifier (from X-Strata-Principal header)
        tenant: Optional team/org identifier (from X-Strata-Tenant header)
        scopes: Set of permission scopes (from X-Strata-Scopes header)
    """

    id: str
    tenant: str | None = None
    scopes: frozenset[str] = field(default_factory=frozenset)

    def has_scope(self, scope: str) -> bool:
        """Check if principal has a specific scope.

        The special scope 'admin:*' grants all permissions.

        Args:
            scope: The scope to check (e.g., 'scan:create', 'admin:cache')

        Returns:
            True if the principal has the scope or admin:* wildcard
        """
        if "admin:*" in self.scopes:
            return True
        return scope in self.scopes


@dataclass(frozen=True)
class TableRef:
    """Canonical table reference for ACL matching.

    Provides a normalized representation of a table for access control
    pattern matching. Format: {catalog}:{namespace}.{table}

    Examples:
        - file:integration.events
        - s3:analytics.clicks

    Attributes:
        catalog: Storage type ('file' or 's3')
        namespace: Database/schema namespace
        table: Table name
    """

    catalog: str  # "file" or "s3"
    namespace: str
    table: str

    @classmethod
    def from_table_identity(
        cls, identity: "TableIdentity", table_uri: str | None = None
    ) -> "TableRef":
        """Convert TableIdentity to canonical TableRef.

        Args:
            identity: TableIdentity from the planner
            table_uri: Original table URI (used to determine catalog type)

        Returns:
            TableRef with normalized catalog, namespace, and table
        """
        # Determine catalog type from URI prefix
        catalog = "file"
        if table_uri and table_uri.startswith("s3://"):
            catalog = "s3"

        return cls(
            catalog=catalog,
            namespace=identity.namespace,
            table=identity.table,
        )

    def __str__(self) -> str:
        """Return canonical string for ACL pattern matching."""
        return f"{self.catalog}:{self.namespace}.{self.table}"


@dataclass(frozen=True)
class TableIdentity:
    """Canonical table identity for deterministic cache keys and metrics.

    Production systems hate ambiguity. This class provides a single,
    canonical way to identify a table regardless of how the user
    specified it (via URI, catalog reference, etc.).

    Format: <catalog>.<namespace>.<table>

    Examples:
        - strata.test_db.events
        - default.analytics.page_views
    """

    catalog: str
    namespace: str
    table: str

    def __str__(self) -> str:
        """Return the canonical string representation."""
        return f"{self.catalog}.{self.namespace}.{self.table}"

    @classmethod
    def from_table_id(cls, table_id: str, catalog: str = "strata") -> "TableIdentity":
        """Create from a table_id like 'namespace.table'.

        Args:
            table_id: Table identifier in format 'namespace.table'
            catalog: Catalog name (default: 'strata')

        Returns:
            TableIdentity with canonical representation
        """
        parts = table_id.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid table_id '{table_id}': expected 'namespace.table' format")
        return cls(catalog=catalog, namespace=parts[0], table=parts[1])


def _wrap_filter_value(value: FilterValue):
    """Wrap a ``FilterValue`` into a pyiceberg ``Literal``.

    ``iceberg_literal`` is generic over a constrained TypeVar — a union
    input doesn't typecheck because the constraint resolver can't pick
    a single slot. Dispatching on isinstance narrows ``value`` to one
    constraint at each call site.
    """
    from pyiceberg.expressions.literals import literal as iceberg_literal

    if isinstance(value, str):
        return iceberg_literal(value)
    if isinstance(value, bool):
        return iceberg_literal(value)
    if isinstance(value, int):
        return iceberg_literal(value)
    if isinstance(value, float):
        return iceberg_literal(value)
    if isinstance(value, bytes):
        return iceberg_literal(value)
    if isinstance(value, uuid.UUID):
        return iceberg_literal(value)
    if isinstance(value, Decimal):
        return iceberg_literal(value)
    if isinstance(value, datetime):
        return iceberg_literal(value)
    if isinstance(value, date):
        return iceberg_literal(value)
    return iceberg_literal(value)


def filters_to_iceberg_expression(filters: list[Filter] | None):
    """Convert Strata filters to a PyIceberg boolean expression.

    Only converts filters that PyIceberg can handle (flat columns, basic ops).
    Uses AND semantics to combine multiple filters.

    Args:
        filters: List of Filter objects

    Returns:
        PyIceberg BooleanExpression, or None if no filters
    """
    if not filters:
        return None

    from functools import reduce

    from pyiceberg.expressions import (
        And,
        EqualTo,
        GreaterThan,
        GreaterThanOrEqual,
        LessThan,
        LessThanOrEqual,
        NotEqualTo,
        Reference,
    )
    from pyiceberg.expressions.literals import Literal as IcebergLiteral

    exprs = []
    for f in filters:
        # Skip nested column references (contain dots)
        if "." in f.column:
            continue

        # Wrap the column name as an ``UnboundTerm`` (``Reference``) and
        # narrow the value into a typed pyiceberg ``Literal`` via a
        # per-branch isinstance dispatch. Pyiceberg accepts raw scalars
        # at runtime via internal coercion, but its constructors are
        # typed to require these wrappers — and ``iceberg_literal`` itself
        # is generic over a *constrained* type variable, so a single
        # call with a ``FilterValue`` union won't typecheck either.
        term = Reference(f.column)
        value: IcebergLiteral = _wrap_filter_value(f.value)
        match f.op:
            case FilterOp.EQ:
                exprs.append(EqualTo(term=term, value=value))
            case FilterOp.NE:
                exprs.append(NotEqualTo(term=term, value=value))
            case FilterOp.LT:
                exprs.append(LessThan(term=term, value=value))
            case FilterOp.LE:
                exprs.append(LessThanOrEqual(term=term, value=value))
            case FilterOp.GT:
                exprs.append(GreaterThan(term=term, value=value))
            case FilterOp.GE:
                exprs.append(GreaterThanOrEqual(term=term, value=value))

    if not exprs:
        return None

    # Combine with AND
    return reduce(And, exprs)


class CacheGranularity(Enum):
    """Cache granularity options.

    Controls what is included in the cache key:

    - ROW_GROUP_PROJECTION (default): Cache key includes row_group_id + projection
      Finest granularity. Different column selections get separate cache entries.
      Best for: Workloads with consistent projections, memory-constrained systems.

    - ROW_GROUP: Cache key includes row_group_id only (ignores projection)
      Same row group = same cache entry regardless of columns requested.
      The cache stores the full row group and projects on readback when needed.
      Best for: Workloads that always request all columns, or when cache reuse
      across different projections is more valuable than projection efficiency.
    """

    ROW_GROUP_PROJECTION = "row_group_projection"
    ROW_GROUP = "row_group"


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key for a row group.

    v2 Cache Key Contract (multi-tenancy):
    - Key format: {tenant_id}|{table_identity}|{snapshot_id}|{file_path}|
      {row_group_id}[|{projection}]
    - Hash: SHA-256 of the key string
    - Storage: cache_dir/v{VERSION}/{tenant_prefix}/{hash[:2]}/{hash[2:4]}/{hash}.arrowstream
    - Version is baked into path (see cache.CACHE_VERSION)

    Tenant isolation:
    - Each tenant's cache entries are stored under tenant-prefixed directories
    - tenant_id is included in the hash to prevent cache key collisions
    - Default tenant "_default" for backward compatibility

    Uses TableIdentity (e.g., 'catalog.namespace.table') instead of the
    user-supplied URI to ensure cache hits work regardless of URI format
    variations like:
    - file:///path#ns.table vs /path#ns.table
    - Different hostnames pointing to same data

    The canonical identity prevents subtle cache duplication bugs where
    the same data is cached multiple times under different keys.

    Cache granularity is controlled by whether projection_fingerprint
    is included in the key hash:
    - ROW_GROUP_PROJECTION: includes projection (default)
    - ROW_GROUP: ignores projection (caches all columns)
    """

    tenant_id: str  # Tenant identifier for cache isolation
    table_identity: TableIdentity  # Canonical identity like 'strata.namespace.table'
    snapshot_id: int
    file_path: str
    row_group_id: int
    projection_fingerprint: str  # Used only if granularity includes projection

    @property
    def table_id(self) -> str:
        """Return canonical table identity string for compatibility."""
        return str(self.table_identity)

    def to_hex(self, granularity: CacheGranularity = CacheGranularity.ROW_GROUP_PROJECTION) -> str:
        """Generate a hex digest for filesystem storage.

        Args:
            granularity: Controls what is included in the hash.
                - ROW_GROUP_PROJECTION: includes projection fingerprint (default)
                - ROW_GROUP: excludes projection, cache stores all columns
        """
        if granularity == CacheGranularity.ROW_GROUP:
            # Ignore projection - cache all columns
            key_str = (
                f"{self.tenant_id}|{self.table_identity}|{self.snapshot_id}|"
                f"{self.file_path}|{self.row_group_id}"
            )
        else:
            # Include projection in key
            key_str = (
                f"{self.tenant_id}|{self.table_identity}|{self.snapshot_id}|"
                f"{self.file_path}|{self.row_group_id}|{self.projection_fingerprint}"
            )
        return hashlib.sha256(key_str.encode()).hexdigest()

    @staticmethod
    def compute_projection_fingerprint(columns: list[str] | None) -> str:
        """Compute a fingerprint for the column projection.

        Column order is preserved in the fingerprint because it matters for
        consumers who expect columns in a specific order.
        """
        if columns is None:
            return "*"
        return hashlib.sha256(",".join(columns).encode()).hexdigest()[:16]


@dataclass
class Task:
    """A single read task for one row group."""

    file_path: str
    row_group_id: int
    cache_key: CacheKey
    num_rows: int
    columns: list[str] | None = None

    # Estimated size from Parquet metadata (for pre-flight checks)
    estimated_bytes: int = 0

    # Populated after fetch
    cached: bool = False
    bytes_read: int = 0


@dataclass
class ReadPlan:
    """A plan for reading data from an Iceberg snapshot.

    Note: table_uri is kept for backwards compatibility and debugging,
    but table_identity is the canonical identifier used for cache keys,
    metrics, and logs.
    """

    table_uri: str  # Original user input (for debugging/display only)
    table_identity: TableIdentity  # Canonical identity for cache/metrics/logs
    snapshot_id: int
    tasks: list[Task] = field(default_factory=list)
    columns: list[str] | None = None
    filters: list[Filter] = field(default_factory=list)

    # Schema from Parquet metadata (no IO at query time)
    schema: "pa.Schema | None" = None

    # Unique scan identifier (generated once at creation)
    scan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    # Metrics
    total_row_groups: int = 0
    pruned_row_groups: int = 0
    planning_time_ms: float = 0.0

    # Estimated response size (sum of row group sizes from Parquet metadata)
    # Used for pre-flight size checks before streaming begins
    estimated_bytes: int = 0

    # Prefetched first row group bytes (optimization to reduce TTFB).
    # Set by server after plan is stored; consumed by streaming endpoint.
    # Using bytes | None avoids asyncio.Future which isn't picklable.
    prefetched_first: bytes | None = None

    # Ownership tracking for authorization (set by server when auth_mode=trusted_proxy)
    owner_principal: str | None = None  # Principal ID who created this scan
    owner_tenant: str | None = None  # Tenant of the owner


class ErrorResponse(BaseModel):
    """Standard error response format.

    v1 Error Codes:
    - 400 Bad Request: Invalid table URI, malformed filters, too many row groups
    - 404 Not Found: Scan ID not found
    - 413 Payload Too Large: Response exceeds max_response_bytes
    - 499 Client Closed Request: Client disconnected mid-scan (logged only)
    - 500 Internal Server Error: Unexpected server error
    - 503 Service Unavailable: Server draining or at capacity
    - 504 Gateway Timeout: Planning or scan exceeded timeout
    """

    detail: str  # Human-readable error message
    error_code: str | None = None  # Optional machine-readable code


class WarmRequest(BaseModel):
    """Request to warm the cache for specific tables.

    Preloads row group data into the cache so subsequent queries are fast.
    This is useful for:
    - Warming cache after server restart
    - Preloading data before a batch of dashboards query it
    - Ensuring low latency for critical tables
    """

    tables: list[str]  # Table URIs to warm (e.g., "file:///warehouse#ns.table")
    columns: list[str] | None = None  # Columns to cache (None = all)
    max_row_groups: int | None = None  # Limit row groups per table (None = all)
    concurrent: int = 4  # Max concurrent fetches


class WarmResponse(BaseModel):
    """Response from cache warming operation."""

    tables_warmed: int  # Number of tables processed
    row_groups_cached: int  # Total row groups written to cache
    row_groups_skipped: int  # Already in cache (cache hits)
    bytes_written: int  # Total bytes written to cache
    elapsed_ms: float  # Total time taken
    errors: list[str]  # Any errors encountered (table URI -> error message)


class WarmAsyncRequest(BaseModel):
    """Request to start an async/background cache warming job.

    Similar to WarmRequest but runs in the background and returns a job ID
    for tracking progress.
    """

    tables: list[str]  # Table URIs to warm
    columns: list[str] | None = None  # Columns to cache (None = all)
    snapshot_id: int | None = None  # Specific snapshot (None = current)
    max_row_groups: int | None = None  # Limit row groups per table
    concurrent: int = 4  # Max concurrent fetches
    priority: int = 0  # Higher = more urgent (affects queue order)


class WarmJobStatus(str, Enum):
    """Status of a background warming job."""

    PENDING = "pending"  # Queued, not started
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Finished with errors
    CANCELLED = "cancelled"  # Cancelled by user


class WarmJobProgress(BaseModel):
    """Progress information for a warming job."""

    job_id: str  # Unique job identifier
    status: WarmJobStatus  # Current status
    tables_total: int  # Total tables to warm
    tables_completed: int  # Tables fully warmed
    row_groups_total: int  # Total row groups across all tables
    row_groups_completed: int  # Row groups fetched (cached + skipped)
    row_groups_cached: int  # Row groups written to cache
    row_groups_skipped: int  # Row groups already cached
    bytes_written: int  # Bytes written so far
    started_at: float | None  # Unix timestamp when job started
    completed_at: float | None  # Unix timestamp when job completed
    elapsed_ms: float  # Time elapsed so far
    current_table: str | None  # Table currently being warmed
    errors: list[str]  # Errors encountered


class WarmAsyncResponse(BaseModel):
    """Response when starting an async warming job."""

    job_id: str  # Unique job ID for tracking
    status: WarmJobStatus  # Initial status (pending or running)
    tables_count: int  # Number of tables in the job
    message: str  # Human-readable status message


def _deserialize_value(value: FilterValue) -> FilterValue:
    """Deserialize filter values from JSON."""
    if isinstance(value, str) and value.startswith("__datetime__:"):
        return datetime.fromisoformat(value.replace("__datetime__:", ""))
    return value


def serialize_filter(f: Filter) -> dict[str, FilterValue]:
    """Serialize a Filter for JSON transport."""
    value = f.value
    if isinstance(value, datetime):
        value = f"__datetime__:{value.isoformat()}"
    return {"column": f.column, "op": f.op.value, "value": value}


# ---------------------------------------------------------------------------
# Unified Materialize API Types
# ---------------------------------------------------------------------------


class TransformSpec(BaseModel):
    """Transform specification for materialize requests.

    Defines the executor and parameters for transforming inputs.
    Built-in executors like scan@v1 are handled by Strata directly.

    Attributes:
        executor: Executor reference (e.g., "scan@v1", "duckdb_sql@v1")
        params: Executor-specific parameters
    """

    executor: str  # "scan@v1", "duckdb_sql@v1", etc.
    params: dict[str, object] = {}  # Executor-specific parameters


class FilterSpec(BaseModel):
    """Row filter specification for identity transform.

    Attributes:
        column: Column name to filter on
        op: Comparison operator
        value: Value to compare against
    """

    column: str
    # Typed as the enum so an invalid operator fails inside ``model_validate``
    # (→ 400) instead of escaping as an uncaught ``ValueError`` from
    # ``FilterOp(f.op)`` later in ``to_strata_filters``.
    op: FilterOp
    value: FilterValue


class IdentityParams(BaseModel):
    """Parameters for the scan@v1 built-in transform.

    The identity transform reads from exactly one Iceberg table input,
    applies optional column projection and row filtering, and returns
    the data unchanged. It's executed internally by Strata (no external
    executor needed).

    Attributes:
        columns: Column projection (None = all columns)
        filters: Row filters (all filters are AND'd together)
        snapshot_id: Specific snapshot to read (None = current)
    """

    columns: list[str] | None = None
    filters: list[FilterSpec] | None = None
    snapshot_id: int | None = None

    def to_strata_filters(self) -> list[Filter]:
        """Convert FilterSpec list to internal Filter objects."""
        if not self.filters:
            return []
        result = []
        for f in self.filters:
            result.append(
                Filter(
                    column=f.column,
                    op=f.op,  # already a FilterOp (validated by the model)
                    value=f.value,
                )
            )
        return result


class MaterializeRequest(BaseModel):
    """Unified request to materialize data.

    This is the single entry point for all data access in Strata.
    Scanning an Iceberg table is expressed as a materialize with
    the scan@v1 transform.

    Attributes:
        inputs: List of input URIs (table URIs or artifact URIs)
        transform: Transform specification (executor + params)
        name: Optional name to assign to the result
        mode: Delivery mode - "stream" for immediate consumption,
              "artifact" for async build with later retrieval
        refresh: Force a fresh build even if a cached artifact exists
        stream_timeout_seconds: Timeout for streaming mode
    """

    inputs: list[str]  # Input URIs: "file:///warehouse#db.events" or "strata://artifact/..."
    transform: TransformSpec
    name: str | None = None  # Optional name to assign (e.g., "daily_revenue")
    mode: str = "stream"  # "stream" | "artifact"
    refresh: bool = False
    stream_timeout_seconds: float | None = None


class MaterializeResponse(BaseModel):
    """Response from materialize request.

    Returns either:
    - Cache hit: artifact exists (state="ready"), can fetch immediately
    - Cache miss: artifact being built (state="building")

    For artifact mode:
    - Client polls /v1/builds/{build_id} or fetches when ready

    For stream mode:
    - Client streams from stream_url while artifact builds in parallel

    Both modes create and persist artifacts. The mode only affects
    how the client receives data, not whether it's cached.

    Attributes:
        hit: True if artifact already existed (cache hit)
        artifact_uri: URI of the artifact (always present)
        state: Current state ("ready", "building")
        build_spec: If miss in personal mode, spec for client to build locally
        build_id: Build ID for polling status (artifact mode, cache miss)
        stream_id: Stream ID for immediate consumption (stream mode)
        stream_url: URL to stream data from (stream mode)
    """

    hit: bool  # True = artifact exists, False = building
    artifact_uri: str  # "strata://artifact/{id}@v={version}"
    state: str = "ready"  # "ready", "building"
    build_spec: dict[str, object] | None = None  # Present if hit=False (personal mode)
    build_id: str | None = None  # Present if hit=False (server/artifact mode)
    stream_id: str | None = None  # Present for stream mode
    stream_url: str | None = None  # "/v1/streams/{stream_id}"


class BuildSpec(BaseModel):
    """Specification for client-side artifact building.

    Returned when materialize() has a cache miss. The client must:
    1. Execute the transform locally using the specified executor
    2. Upload the result via upload_finalize
    3. Optionally set a name pointer

    Attributes:
        artifact_id: ID of the artifact being built
        version: Version number of the artifact
        executor: Executor URI (e.g., "local://duckdb_sql@v1")
        params: Executor-specific parameters
        input_uris: Resolved input URIs (tables or artifacts)
    """

    artifact_id: str
    version: int
    executor: str
    params: dict[str, object]
    input_uris: list[str]  # Resolved URIs for inputs


class UploadFinalizeRequest(BaseModel):
    """Request to finalize an artifact upload.

    After the client builds an artifact locally, it uploads the Arrow IPC
    data and calls this endpoint to finalize the artifact.

    Attributes:
        artifact_id: ID of the artifact being finalized
        version: Version number of the artifact
        arrow_schema: Arrow schema as JSON string
        row_count: Number of rows in the artifact
        name: Optional name to assign after finalization
    """

    artifact_id: str
    version: int
    arrow_schema: str  # Arrow schema serialized as JSON
    row_count: int  # Number of rows in the result
    name: str | None = None  # Optional name to assign


class UploadFinalizeResponse(BaseModel):
    """Response from upload finalization.

    Attributes:
        artifact_uri: Final artifact URI
        byte_size: Size of the stored artifact in bytes
        name_uri: Name URI if a name was assigned
    """

    artifact_uri: str  # "strata://artifact/{id}@v={version}"
    byte_size: int
    name_uri: str | None = None  # "strata://name/{name}" if name was set


class PutArtifactRequest(BaseModel):
    """Request to directly upload and persist an artifact.

    This is a simplified API for clients that execute transforms locally
    and want to persist the result with full provenance tracking.
    Supports deduplication: if an artifact with the same provenance hash
    already exists, returns the existing artifact (cache hit).

    Attributes:
        inputs: List of input URIs (artifact URIs or table URIs)
        transform: Transform specification (executor + params)
            The params are opaque to Strata and used only for provenance.
        data: The artifact data as JSON (will be stored as Arrow)
        name: Optional name to assign to the artifact
    """

    inputs: list[str]  # Input URIs: "strata://artifact/..." or "file:///..."
    transform: TransformSpec  # Opaque transform spec for provenance
    data: dict[str, object]  # JSON data to persist
    name: str | None = None  # Optional name pointer


class PutArtifactResponse(BaseModel):
    """Response from put artifact request.

    Attributes:
        artifact_uri: The artifact URI
        hit: True if this was a cache hit (existing artifact)
        byte_size: Size of the stored artifact in bytes
        name_uri: Name URI if a name was assigned
    """

    artifact_uri: str  # "strata://artifact/{id}@v={version}"
    hit: bool  # True if deduplicated to existing artifact
    byte_size: int
    name_uri: str | None = None  # "strata://name/{name}" if name was set


class NameResolveRequest(BaseModel):
    """Request to resolve a name to an artifact.

    Attributes:
        name: Name to resolve (without strata://name/ prefix)
    """

    name: str


class NameResolveResponse(BaseModel):
    """Response from name resolution.

    Attributes:
        artifact_uri: Resolved artifact URI
        version: Pinned version
        updated_at: Timestamp of last name update
    """

    artifact_uri: str  # "strata://artifact/{id}@v={version}"
    version: int
    updated_at: float  # Unix timestamp


class NameSetRequest(BaseModel):
    """Request to set or update a name pointer.

    Attributes:
        name: Name to set
        artifact_id: Target artifact ID
        version: Target version
    """

    name: str
    artifact_id: str
    version: int


class NameSetResponse(BaseModel):
    """Response from setting a name.

    Attributes:
        name_uri: URI of the name pointer
        artifact_uri: URI of the target artifact
    """

    name_uri: str  # "strata://name/{name}"
    artifact_uri: str  # "strata://artifact/{id}@v={version}"


class ArtifactInfoResponse(BaseModel):
    """Response with artifact metadata.

    Attributes:
        artifact_id: Artifact ID
        version: Version number
        state: Lifecycle state ("building", "ready", "failed")
        arrow_schema: Arrow schema as JSON (if ready)
        row_count: Number of rows (if ready)
        byte_size: Size in bytes (if ready)
        created_at: Creation timestamp
    """

    artifact_id: str
    version: int
    state: str
    arrow_schema: str | None = None
    row_count: int | None = None
    byte_size: int | None = None
    created_at: float


class InputChangeInfo(BaseModel):
    """Information about a changed input dependency.

    Attributes:
        input_uri: The input URI that changed
        old_version: The version used when artifact was built
        new_version: The current version of the input
    """

    input_uri: str
    old_version: str
    new_version: str


class NameStatusResponse(BaseModel):
    """Response with named artifact status including staleness info.

    Use GET /v1/artifacts/names/{name}/status to get this information.

    Attributes:
        name: The artifact name
        artifact_uri: URI of the pinned artifact version
        artifact_id: Artifact ID
        version: Pinned version number
        state: Artifact state ("ready", "building", "failed")
        updated_at: When the name was last updated
        input_versions: Mapping of input URI -> version when built
        is_stale: True if any input has changed since build
        stale_reason: Human-readable explanation if stale
        changed_inputs: List of inputs that have newer versions
    """

    name: str
    artifact_uri: str
    artifact_id: str
    version: int
    state: str
    updated_at: float
    input_versions: dict[str, str]
    is_stale: bool = False
    stale_reason: str | None = None
    changed_inputs: list[InputChangeInfo] | None = None


class BuildProgress(BaseModel):
    """Progress information for an in-progress build.

    Provides optional progress metrics for builds that support them.
    For identity transforms, this tracks bytes processed from Parquet files.

    Attributes:
        bytes_processed: Bytes read/processed so far
        estimated_total_bytes: Estimated total bytes to process
        rows_processed: Rows processed so far (if known)
        estimated_total_rows: Estimated total rows (if known)
    """

    bytes_processed: int = 0
    estimated_total_bytes: int | None = None
    rows_processed: int | None = None
    estimated_total_rows: int | None = None


class BuildStatusResponse(BaseModel):
    """Response with async build status for server-mode transforms.

    Use GET /v1/artifacts/builds/{build_id} to poll build status.
    A compatibility alias also exists at GET /v1/builds/{build_id}.

    Attributes:
        build_id: Unique build identifier
        artifact_id: Target artifact ID
        version: Target artifact version
        state: Current state (pending, building, ready, failed)
        artifact_uri: URI of the artifact (available when state=ready)
        executor_ref: Executor reference
        progress: Optional progress information (while building)
        created_at: When the build was created
        started_at: When execution started (if started)
        completed_at: When execution finished (if finished)
        error_message: Error details (if failed)
        error_code: Error code for programmatic handling (if failed)
    """

    build_id: str
    artifact_id: str
    version: int
    state: str  # "pending", "building", "ready", "failed"
    artifact_uri: str
    executor_ref: str
    progress: BuildProgress | None = None  # Present while state="building"
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    error_code: str | None = None


class ExplainMaterializeRequest(BaseModel):
    """Request to explain what materialize would do (dry run).

    Attributes:
        inputs: List of input URIs (table URIs or artifact URIs)
        transform: Transform specification (executor + params)
        name: Optional name to check against existing artifact
    """

    inputs: list[str]
    transform: TransformSpec
    name: str | None = None


class ExplainMaterializeResponse(BaseModel):
    """Response explaining what materialize would do.

    This is a dry-run that doesn't modify anything but shows:
    - Whether the result would be a cache hit or miss
    - If checking a name, whether it's stale
    - Which inputs have changed if stale

    Attributes:
        would_hit: True if materialize would return a cached artifact
        artifact_uri: URI of existing artifact (if would_hit) or None
        would_build: True if client would need to build locally
        is_stale: True if named artifact exists but inputs have changed
        stale_reason: Explanation of why rebuild is needed
        changed_inputs: List of inputs that changed since last build
        resolved_input_versions: Current versions of all inputs
    """

    would_hit: bool
    artifact_uri: str | None = None
    would_build: bool = False
    is_stale: bool = False
    stale_reason: str | None = None
    changed_inputs: list[InputChangeInfo] | None = None
    resolved_input_versions: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Lineage and Dependency Introspection
# ---------------------------------------------------------------------------


class LineageNode(BaseModel):
    """A node in the artifact lineage graph.

    Attributes:
        uri: Artifact URI (strata://artifact/{id}@v={version}) or table URI
        artifact_id: Artifact ID (if this is an artifact, not a table)
        version: Artifact version (if this is an artifact)
        type: "artifact" or "table"
        transform_ref: Transform executor reference (if artifact)
        created_at: When artifact was created (if artifact)
    """

    uri: str
    artifact_id: str | None = None
    version: int | None = None
    type: str  # "artifact" | "table"
    transform_ref: str | None = None
    created_at: float | None = None


class LineageEdge(BaseModel):
    """An edge in the artifact lineage graph (input dependency).

    Attributes:
        from_uri: Source URI (the input)
        to_uri: Target URI (the artifact that uses this input)
        input_version: Version string of the input when used
    """

    from_uri: str
    to_uri: str
    input_version: str


class ArtifactLineageResponse(BaseModel):
    """Response with artifact lineage (input dependency graph).

    Shows the full input dependency tree for an artifact, including:
    - Direct inputs (tables and artifacts)
    - Transitive inputs (inputs of input artifacts, recursively)

    Use GET /v1/artifacts/{artifact_id}/v/{version}/lineage to get this.

    Attributes:
        artifact_uri: The artifact being queried
        artifact_id: Artifact ID
        version: Artifact version
        nodes: All nodes in the lineage graph (artifacts and tables)
        edges: All edges (input relationships) in the graph
        depth: Maximum depth of the lineage tree
        direct_inputs: URIs of direct inputs (first-level dependencies)
    """

    artifact_uri: str
    artifact_id: str
    version: int
    nodes: list[LineageNode]
    edges: list[LineageEdge]
    depth: int
    direct_inputs: list[str]


class DependentInfo(BaseModel):
    """Information about an artifact that depends on another.

    Attributes:
        artifact_uri: URI of the dependent artifact
        artifact_id: Artifact ID
        version: Artifact version
        name: Name pointing to this artifact (if any)
        transform_ref: Transform executor reference
        created_at: When the dependent artifact was created
        input_version: Version string this artifact uses for the dependency
    """

    artifact_uri: str
    artifact_id: str
    version: int
    name: str | None = None
    transform_ref: str | None = None
    created_at: float | None = None
    input_version: str


class ArtifactDependentsResponse(BaseModel):
    """Response with artifacts that depend on a given artifact.

    Shows reverse dependencies: which artifacts use this artifact as input.
    Useful for impact analysis when considering artifact changes/deletion.

    Use GET /v1/artifacts/{artifact_id}/v/{version}/dependents to get this.

    Attributes:
        artifact_uri: The artifact being queried
        artifact_id: Artifact ID
        version: Artifact version
        dependents: List of artifacts that use this artifact as input
        total_count: Total number of dependents found
    """

    artifact_uri: str
    artifact_id: str
    version: int
    dependents: list[DependentInfo]
    total_count: int


# ---------------------------------------------------------------------------
# Executor Protocol v1 - Stable interface for external executors
# ---------------------------------------------------------------------------
#
# Protocol Version: v1
# Header: X-Strata-Executor-Protocol: v1
#
# This defines the stable interface for implementing Strata executors.
# Executors receive Arrow IPC inputs, run a transform, and return Arrow IPC output.
#
# Push Model (Strata streams to executor):
#   POST {executor_url}/v1/execute
#   Content-Type: multipart/form-data
#   X-Strata-Executor-Protocol: v1
#
# Pull Model (Executor pulls from Strata):
#   GET /v1/builds/{build_id}/manifest -> ExecutorManifest
#   Executor downloads inputs, executes, uploads output, calls finalize
#
# ---------------------------------------------------------------------------

# Protocol version constant
EXECUTOR_PROTOCOL_VERSION = "v1"

# HTTP headers for protocol versioning
EXECUTOR_PROTOCOL_HEADER = "X-Strata-Executor-Protocol"
EXECUTOR_LOGS_HEADER = "X-Strata-Logs"


class ExecutorInputDescriptor(BaseModel):
    """Descriptor for a single input in an executor request.

    Attributes:
        name: Input name (e.g., "input0", "input1")
        format: Data format (always "arrow_ipc_stream" in v1)
        uri: Original input URI (for debugging/logging)
        byte_size: Size of the input in bytes (if known)
    """

    name: str
    format: str = "arrow_ipc_stream"
    uri: str | None = None
    byte_size: int | None = None


class ExecutorTransformSpec(BaseModel):
    """Transform specification sent to executor.

    Attributes:
        ref: Transform reference (e.g., "duckdb_sql@v1")
        code_hash: Hash of the transform code (for reproducibility)
        params: Executor-specific parameters (e.g., {"sql": "SELECT ..."})
    """

    ref: str
    code_hash: str
    params: dict[str, object]


class ExecutorRequestMetadata(BaseModel):
    """Metadata sent to executor in push model requests.

    This is the JSON payload in the "metadata" part of the multipart request.
    It provides all context needed for the executor to run the transform.

    Attributes:
        protocol_version: Protocol version (always "v1" for this schema)
        build_id: Unique build identifier for tracing/logging
        tenant: Tenant ID (for multi-tenant deployments)
        principal: Principal ID who initiated the build
        provenance_hash: Hash of inputs + transform for deduplication
        transform: Transform specification with ref, code_hash, params
        inputs: List of input descriptors (name, format, uri)
    """

    protocol_version: str = EXECUTOR_PROTOCOL_VERSION
    build_id: str
    tenant: str | None = None
    principal: str | None = None
    provenance_hash: str
    transform: ExecutorTransformSpec
    inputs: list[ExecutorInputDescriptor]


class ExecutorResponse(BaseModel):
    """Response from executor (for structured error responses).

    Success responses return Arrow IPC stream directly with 200 status.
    Error responses return JSON with this structure.

    Attributes:
        success: Whether the execution succeeded
        error_code: Machine-readable error code
        error_message: Human-readable error message
        duration_ms: Execution time in milliseconds
        output_rows: Number of rows in output (on success)
        output_bytes: Size of output in bytes (on success)
        logs: Executor logs (stderr/stdout)
    """

    success: bool
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None
    output_rows: int | None = None
    output_bytes: int | None = None
    logs: str | None = None


class ExecutorManifestInput(BaseModel):
    """Input descriptor in pull model manifest.

    Attributes:
        name: Input name (e.g., "input0")
        download_url: Signed URL to download the input
        byte_size: Expected size in bytes
        format: Data format (always "arrow_ipc_stream")
    """

    name: str
    download_url: str
    byte_size: int | None = None
    format: str = "arrow_ipc_stream"


class ExecutorManifest(BaseModel):
    """Manifest returned for pull model execution.

    The executor uses this manifest to:
    1. Download inputs from signed URLs
    2. Execute the transform
    3. Upload output to the signed URL
    4. Call finalize_url to complete the build

    Attributes:
        protocol_version: Protocol version (always "v1")
        build_id: Unique build identifier
        metadata: Transform metadata (ref, params, etc.)
        inputs: List of inputs with signed download URLs
        upload_url: Signed URL to upload the output
        finalize_url: URL to call after upload completes
        max_output_bytes: Maximum allowed output size
        timeout_seconds: Maximum execution time
    """

    protocol_version: str = EXECUTOR_PROTOCOL_VERSION
    build_id: str
    metadata: dict[str, object]
    inputs: list[ExecutorManifestInput]
    upload_url: str
    finalize_url: str
    max_output_bytes: int
    timeout_seconds: float


class ExecutorCapabilities(BaseModel):
    """Executor capabilities reported in health check.

    Executors should return this from GET /health to describe their capabilities.

    Attributes:
        protocol_versions: List of supported protocol versions
        transform_refs: List of supported transform references
        max_input_bytes: Maximum total input size supported
        max_output_bytes: Maximum output size supported
        max_concurrent_executions: Maximum concurrent executions
        features: Optional feature flags (e.g., {"streaming": true})
    """

    protocol_versions: list[str] = [EXECUTOR_PROTOCOL_VERSION]
    transform_refs: list[str] = []
    max_input_bytes: int | None = None
    max_output_bytes: int | None = None
    max_concurrent_executions: int | None = None
    features: dict[str, object] | None = None


class ExecutorHealthResponse(BaseModel):
    """Health check response from executor.

    GET /health should return this structure.

    Attributes:
        status: "healthy", "degraded", or "unhealthy"
        capabilities: Executor capabilities
        version: Executor software version
        uptime_seconds: Seconds since executor started
        active_executions: Current number of active executions
    """

    status: str  # "healthy" | "degraded" | "unhealthy"
    capabilities: ExecutorCapabilities
    version: str | None = None
    uptime_seconds: float | None = None
    active_executions: int | None = None
