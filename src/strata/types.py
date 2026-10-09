"""Core types for Strata."""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

# Filter types live in dependency-free ``strata.filters`` so the client avoids pydantic;
# re-exported so ``from strata.types import Filter`` keeps working.
from strata.filters import (  # noqa: F401
    Filter,
    FilterOp,
    FilterValue,
    SupportsOrdering,
    compute_filter_fingerprint,
    deserialize_filter_value,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from strata.iceberg_schema import Column
    from strata.metadata_cache import EqualityDeleteEntry


# --- Authentication / Authorization Types ---


@dataclass(frozen=True)
class Principal:
    """Authenticated caller identity, from trusted-proxy headers or an API key.

    Under trusted-proxy auth Strata does not authenticate; it trusts
    ``X-Strata-Principal``, the configured tenant header and ``X-Strata-Scopes``.
    """

    id: str
    tenant: str | None = None
    scopes: frozenset[str] = field(default_factory=frozenset)

    def has_scope(self, scope: str) -> bool:
        """Return True if the principal holds ``scope``; ``admin:*`` grants every scope."""
        if "admin:*" in self.scopes:
            return True
        return scope in self.scopes


@dataclass(frozen=True)
class TableRef:
    """Canonical table reference for ACL matching, rendered ``{catalog}:{namespace}.{table}``.

    ``catalog`` is a configured catalog's name, or the warehouse store (``file``,
    ``s3``, ``gs``, ``az``), e.g. ``s3:analytics.clicks``.
    """

    catalog: str  # "file" or "s3"
    namespace: str
    table: str

    @classmethod
    def from_table_identity(
        cls,
        identity: "TableIdentity",
        table_uri: str | None = None,
        named_catalog_name: str | None = None,
    ) -> "TableRef":
        """Build the ACL reference for a planner identity.

        A configured catalog is named by ``named_catalog_name``; otherwise the
        store comes from ``table_uri``'s scheme, so a local rule cannot grant a
        same-named bucket table.
        """
        # A configured catalog is named as ACL rules name it ("lake:taxi.*"); a warehouse
        # URI by its store, so a rule for a local table cannot grant a same-named bucket table.
        catalog = "file"
        if named_catalog_name:
            catalog = named_catalog_name
        elif table_uri:
            for prefix, store in _ACL_STORES:
                if table_uri.startswith(prefix):
                    catalog = store
                    break

        return cls(
            catalog=catalog,
            namespace=identity.namespace,
            table=identity.table,
        )

    def __str__(self) -> str:
        """Return canonical string for ACL pattern matching."""
        return f"{self.catalog}:{self.namespace}.{self.table}"


# The store an ACL rule names a warehouse table by. A scheme-less URI is a local path ("file").
_ACL_STORES = (
    ("s3://", "s3"),
    ("gs://", "gs"),
    ("gcs://", "gs"),
    ("abfss://", "az"),
    ("abfs://", "az"),
    ("az://", "az"),
    ("azure://", "az"),
)

ACL_STORE_NAMES = ("file", *dict.fromkeys(store for _, store in _ACL_STORES))


@dataclass(frozen=True)
class TableIdentity:
    """Canonical ``<catalog>.<namespace>.<table>`` identity for cache keys and metrics.

    The same table yields the same identity however the URI was spelled.
    """

    catalog: str
    namespace: str
    table: str

    def __str__(self) -> str:
        """Return the canonical string representation."""
        return f"{self.catalog}.{self.namespace}.{self.table}"

    @classmethod
    def from_table_id(cls, table_id: str, catalog: str = "strata") -> "TableIdentity":
        """Create from a ``'namespace.table'`` id; raises ValueError on any other shape."""
        parts = table_id.split(".")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid table_id '{table_id}': expected 'namespace.table' format")
        return cls(catalog=catalog, namespace=parts[0], table=parts[1])


def _wrap_filter_value(value: FilterValue):
    """Wrap a ``FilterValue`` into a pyiceberg ``Literal``.

    The isinstance chain narrows the union per call, since ``iceberg_literal``'s
    constrained TypeVar cannot accept a union.
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
    """AND Strata filters into a PyIceberg expression, or None if none apply.

    Nested (dotted) columns are skipped, which only widens the scan.
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

        # pyiceberg's constructors are typed to require a ``Reference`` and a typed ``Literal``
        # (raw scalars only coerce at runtime), and ``iceberg_literal`` is generic over a
        # constrained TypeVar, so a ``FilterValue`` union must be narrowed per branch.
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

    return reduce(And, exprs)


class CacheGranularity(Enum):
    """Whether the cache key includes the projection.

    ``ROW_GROUP_PROJECTION`` (default) caches each column selection separately.
    ``ROW_GROUP`` caches the full row group and projects on read, trading size
    for reuse across projections.
    """

    ROW_GROUP_PROJECTION = "row_group_projection"
    ROW_GROUP = "row_group"


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key for one row group.

    SHA-256 over ``tenant|table_identity|snapshot|file|row_group[|projection]|schema=``;
    the projection is included only under ``ROW_GROUP_PROJECTION``. Keyed by the
    canonical ``TableIdentity``, not the URI, so URI spellings share entries.
    """

    tenant_id: str
    table_identity: TableIdentity  # Canonical identity like 'strata.namespace.table'
    snapshot_id: int
    file_path: str
    row_group_id: int
    projection_fingerprint: str  # Used only if granularity includes projection
    # The Iceberg schema the row group was read as: a schema change makes no
    # snapshot, so the snapshot alone cannot say.
    schema_id: int | None = None

    @property
    def table_id(self) -> str:
        """Return the canonical table identity string."""
        return str(self.table_identity)

    def to_hex(self, granularity: CacheGranularity = CacheGranularity.ROW_GROUP_PROJECTION) -> str:
        """Return the key's SHA-256 hex digest; ``granularity`` decides if projection counts."""
        if granularity == CacheGranularity.ROW_GROUP:
            key_str = (
                f"{self.tenant_id}|{self.table_identity}|{self.snapshot_id}|"
                f"{self.file_path}|{self.row_group_id}"
            )
        else:
            key_str = (
                f"{self.tenant_id}|{self.table_identity}|{self.snapshot_id}|"
                f"{self.file_path}|{self.row_group_id}|{self.projection_fingerprint}"
            )
        key_str += f"|schema={self.schema_id}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    @staticmethod
    def compute_projection_fingerprint(columns: list[str] | None) -> str:
        """Fingerprint a column projection, order-sensitive; ``"*"`` for all columns.

        Hashed as JSON, not joined, so ``["a,b"]`` and ``["a", "b"]`` differ.
        """
        if columns is None:
            return "*"
        return hashlib.sha256(json.dumps(columns).encode()).hexdigest()[:16]


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

    # Row-group-relative positions of rows the snapshot deleted (Iceberg
    # merge-on-read), sorted; the fetcher drops them. ``num_rows`` already
    # excludes them.
    deleted_rows: "pa.Array | None" = None

    # How the file holds the snapshot's columns when its schema predates the
    # snapshot's (Iceberg schema evolution); None when it matches.
    file_columns: "tuple[Column, ...] | None" = None

    # Equality deletes that may remove rows of this row group (merge-on-read
    # by value), and for each key field id the file's column holding it (None
    # when the file predates it). ``num_rows`` is an upper bound when set.
    equality_deletes: "tuple[EqualityDeleteEntry, ...]" = ()
    # (field id, the file's column or None, what the key reads as when the
    # file lacks it: its identity-partition value or v3 initial-default)
    equality_columns: tuple[tuple[int, str | None, Any], ...] = ()
    # Whether nanosecond timestamps read at microseconds, the only unit of an
    # Iceberg v1 or v2 table; equality delete keys are compared so.
    downcast_ns: bool = False

    # Populated after fetch
    cached: bool = False
    bytes_read: int = 0


@dataclass
class ReadPlan:
    """A plan for reading an Iceberg snapshot; caches key on ``table_identity``, not the URI."""

    table_uri: str  # Original user input (for debugging/display only)
    table_identity: TableIdentity  # Canonical identity for cache/metrics/logs
    snapshot_id: int | None  # None: the table has no snapshots, so the plan has no tasks
    tasks: list[Task] = field(default_factory=list)
    columns: list[str] | None = None
    filters: list[Filter] = field(default_factory=list)

    # Schema from Parquet metadata (no IO at query time)
    schema: "pa.Schema | None" = None

    # The Iceberg schema the scan read (a schema change makes no snapshot).
    schema_id: int | None = None

    scan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    total_row_groups: int = 0
    pruned_row_groups: int = 0
    planning_time_ms: float = 0.0

    # Sum of row-group sizes from Parquet metadata, for the pre-flight size check
    estimated_bytes: int = 0

    # First row group, prefetched to cut TTFB; consumed by the stream endpoint.
    # bytes rather than an asyncio.Future so the plan stays picklable.
    prefetched_first: bytes | None = None

    # Set when auth_mode=trusted_proxy, for stream ownership checks
    owner_principal: str | None = None
    owner_tenant: str | None = None


class ErrorResponse(BaseModel):
    """Standard error body.

    Statuses: 400 bad request, 404 unknown scan, 413 over ``max_response_bytes``,
    499 client gone (logged only), 503 draining or at capacity, 504 timeout.
    """

    detail: str
    error_code: str | None = None


class WarmRequest(BaseModel):
    """Request to preload tables' row groups into the cache."""

    tables: list[str]  # Table URIs to warm (e.g., "file:///warehouse#ns.table")
    columns: list[str] | None = None  # Columns to cache (None = all)
    max_row_groups: int | None = Field(default=None, ge=1)  # Per table (None = all)
    # ge=1: 0 would build ``asyncio.Semaphore(0)`` and every fetch would block forever.
    concurrent: int = Field(default=4, ge=1, le=64)  # Max concurrent fetches


class WarmResponse(BaseModel):
    """Response from cache warming operation."""

    tables_warmed: int
    row_groups_cached: int
    row_groups_skipped: int  # Already in cache
    bytes_written: int
    elapsed_ms: float
    errors: list[str]


class WarmAsyncRequest(BaseModel):
    """Request to warm the cache in a background job, tracked by job ID."""

    tables: list[str]
    columns: list[str] | None = None  # None = all
    snapshot_id: int | None = None  # Specific snapshot (None = current)
    max_row_groups: int | None = Field(default=None, ge=1)  # Limit per table
    concurrent: int = Field(default=4, ge=1, le=64)  # Max concurrent fetches
    priority: int = 0  # Higher = more urgent (affects queue order)


class WarmJobStatus(StrEnum):
    """Status of a background warming job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WarmJobProgress(BaseModel):
    """Progress information for a warming job."""

    job_id: str
    status: WarmJobStatus
    tables_total: int
    tables_completed: int
    row_groups_total: int
    row_groups_completed: int  # cached + skipped
    row_groups_cached: int
    row_groups_skipped: int
    bytes_written: int
    started_at: float | None  # Unix timestamp
    completed_at: float | None  # Unix timestamp
    elapsed_ms: float
    current_table: str | None
    errors: list[str]


class WarmAsyncResponse(BaseModel):
    """Response when starting an async warming job."""

    job_id: str
    status: WarmJobStatus  # pending or running
    tables_count: int
    message: str


# --- Unified Materialize API Types ---


class TransformSpec(BaseModel):
    """Executor reference (e.g. ``scan@v1``, ``duckdb_sql@v1``) and its parameters."""

    executor: str  # "scan@v1", "duckdb_sql@v1", etc.
    params: dict[str, object] = {}


class FilterSpec(BaseModel):
    """A pruning hint for a scan. It does NOT filter rows.

    It skips whole files and row groups by their stats, so the result is a
    **superset** of the matching rows; apply the predicate yourself if needed.
    Pruning is conservative and never drops a matching row.
    """

    column: str
    # The enum makes a bad operator fail in ``model_validate`` (400), not as an uncaught
    # ``ValueError`` in ``to_strata_filters``.
    op: FilterOp
    # JSON-native scalars only: clients tag richer types (datetime/Decimal/UUID/bytes) via
    # ``serialize_filter_value`` and ``to_strata_filters`` decodes them. The narrow type
    # also stops pydantic coercing a plain ISO string into a ``datetime``.
    value: str | bool | int | float


class IdentityParams(BaseModel):
    """Parameters for the built-in scan@v1 transform over exactly one Iceberg table.

    ``columns`` None reads all columns; ``filters`` are ANDed pruning hints;
    ``snapshot_id`` None reads the current snapshot.
    """

    columns: list[str] | None = None
    filters: list[FilterSpec] | None = None
    snapshot_id: int | None = None

    def to_strata_filters(self) -> list[Filter]:
        """Convert to Filters, decoding ``serialize_filter_value`` tags back to Python types."""
        if not self.filters:
            return []
        result = []
        for f in self.filters:
            result.append(
                Filter(
                    column=f.column,
                    op=f.op,  # already a FilterOp (validated by the model)
                    value=deserialize_filter_value(f.value),
                )
            )
        return result


class MaterializeRequest(BaseModel):
    """Request to materialize data; a table scan is a materialize with scan@v1.

    ``mode`` is ``"stream"`` (consume now) or ``"artifact"`` (build, fetch later);
    ``refresh`` forces a build despite a cached artifact.
    """

    inputs: list[str]  # Input URIs: "file:///warehouse#db.events" or "strata://artifact/..."
    transform: TransformSpec
    name: str | None = None  # e.g. "daily_revenue"
    mode: str = "stream"  # "stream" | "artifact"
    refresh: bool = False
    stream_timeout_seconds: float | None = None


class MaterializeResponse(BaseModel):
    """Result of a materialize: a ready hit, or a miss that is ``building``.

    On a miss, artifact mode polls ``build_id`` and stream mode reads ``stream_url``
    while the artifact persists; both modes cache the result.
    """

    hit: bool  # True = artifact exists, False = building
    artifact_uri: str  # "strata://artifact/{id}@v={version}"
    state: str = "ready"  # "ready", "building"
    build_spec: dict[str, object] | None = None  # Present if hit=False (personal mode)
    build_id: str | None = None  # Present if hit=False (server/artifact mode)
    stream_id: str | None = None  # Present for stream mode
    stream_url: str | None = None  # "/v1/streams/{stream_id}"


class BuildSpec(BaseModel):
    """What a client must build locally after a miss, then upload via upload_finalize."""

    artifact_id: str
    version: int
    executor: str
    params: dict[str, object]
    input_uris: list[str]


class UploadFinalizeRequest(BaseModel):
    """Request to finalize a locally built artifact after its Arrow IPC upload."""

    artifact_id: str
    version: int
    arrow_schema: str  # Arrow schema serialized as JSON
    row_count: int
    name: str | None = None


class UploadFinalizeResponse(BaseModel):
    """Response from upload finalization; ``name_uri`` is set only if a name was assigned."""

    artifact_uri: str  # "strata://artifact/{id}@v={version}"
    byte_size: int
    name_uri: str | None = None  # "strata://name/{name}" if name was set


class PutArtifactRequest(BaseModel):
    """Request to persist a locally computed artifact with provenance.

    An existing artifact with the same provenance hash is returned instead.
    ``transform.params`` are opaque, used only for provenance; ``data`` is JSON
    stored as Arrow.
    """

    inputs: list[str]  # Input URIs: "strata://artifact/..." or "file:///..."
    transform: TransformSpec  # Opaque transform spec for provenance
    data: dict[str, object]
    name: str | None = None


class PutArtifactResponse(BaseModel):
    """Response from a put; ``hit`` means an existing artifact was returned."""

    artifact_uri: str  # "strata://artifact/{id}@v={version}"
    hit: bool  # True if deduplicated to existing artifact
    byte_size: int
    name_uri: str | None = None  # "strata://name/{name}" if name was set


class NameResolveRequest(BaseModel):
    """Request to resolve a name (without the ``strata://name/`` prefix) to an artifact."""

    name: str


class NameResolveResponse(BaseModel):
    """Response from name resolution."""

    artifact_uri: str  # "strata://artifact/{id}@v={version}"
    version: int
    updated_at: float  # Unix timestamp


class NameSetRequest(BaseModel):
    """Request to set or update a name pointer."""

    name: str
    artifact_id: str
    version: int


class NameSetResponse(BaseModel):
    """Response from setting a name."""

    name_uri: str  # "strata://name/{name}"
    artifact_uri: str  # "strata://artifact/{id}@v={version}"


class ArtifactInfoResponse(BaseModel):
    """Artifact metadata; schema, row count and size are set once ready.

    Attributes:
        content_sha256: Digest of the stored bytes, to compare outputs without
            downloading; None when never recorded.
        provenance_hash: The dedup key, which a store copying the artifact must keep.
        transform_spec: Stored spec as JSON; ``params.content_type`` says how to read the bytes.
        input_versions: Stored ``input URI -> version`` map as JSON, the edges lineage walks.
    """

    artifact_id: str
    version: int
    state: str
    arrow_schema: str | None = None
    row_count: int | None = None
    byte_size: int | None = None
    created_at: float
    content_sha256: str | None = None
    provenance_hash: str | None = None
    transform_spec: str | None = None
    input_versions: str | None = None


#: Marks a by-provenance 404 as a real miss. Other 404s (a server without the route, a
#: gateway without a store) would otherwise read as a miss and recompute forever.
#: ``strata_client`` keeps its own copy: the packages are deliberately independent.
PROVENANCE_MISS_HEADER = "X-Strata-Provenance-Miss"


class ArtifactProvenanceMatchResponse(BaseModel):
    """An existing artifact for a provenance hash: a team-cache hit.

    Attributes:
        artifact_id: Storage id; usually not one the caller would construct, since
            the provenance hash is the join key.
        provenance_hash: Echoed so batched lookups can be correlated.
        content_type: From the stored spec, so the blob can be decoded without another
            round trip; empty when unrecorded.
        principal: Who computed it, so the hit can be attributed.
        build_env: Interpreter and platform, e.g. ``cpython-3.14-linux-x86_64``. Not
            hashed into provenance, so hits cross platforms; empty when unrecorded.
        build_duration_ms: How long the producing run took (what the hit saved);
            0 when unrecorded.
        env_hash: Digest of lockfiles and runtime env, part of the provenance key;
            with ``build_env``, explains why one caller hits and another misses.
        content_sha256: Digest of the stored bytes; None when unrecorded.
        promotion: Name of the promotion that brought the row here; None for a
            cache publish.
    """

    artifact_id: str
    version: int
    provenance_hash: str
    content_type: str = ""
    state: str = "ready"
    arrow_schema: str | None = None
    row_count: int | None = None
    byte_size: int | None = None
    created_at: float = 0.0
    principal: str | None = None
    build_env: str = ""
    build_duration_ms: int = 0
    env_hash: str = ""
    promotion: str | None = None
    content_sha256: str | None = None


class InputChangeInfo(BaseModel):
    """An input whose version changed since the artifact was built."""

    input_uri: str
    old_version: str
    new_version: str


class NameStatusResponse(BaseModel):
    """A named artifact's status; ``is_stale`` when any input changed since the build."""

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
    """Optional progress of a running build (bytes read from Parquet, for scans)."""

    bytes_processed: int = 0
    estimated_total_bytes: int | None = None
    rows_processed: int | None = None
    estimated_total_rows: int | None = None


class BuildStatusResponse(BaseModel):
    """Status of a server-side build, polled at ``GET /v1/artifacts/builds/{build_id}``.

    ``state`` is pending, building, ready or failed; error fields are set only on failure.
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
    """Dry-run materialize; ``name`` is checked for staleness against its artifact."""

    inputs: list[str]
    transform: TransformSpec
    name: str | None = None


class ExplainMaterializeResponse(BaseModel):
    """What materialize would do, without changing anything: hit or build, and staleness."""

    would_hit: bool
    artifact_uri: str | None = None
    would_build: bool = False
    is_stale: bool = False
    stale_reason: str | None = None
    changed_inputs: list[InputChangeInfo] | None = None
    resolved_input_versions: dict[str, str] | None = None


# --- Lineage and Dependency Introspection ---


class LineageNode(BaseModel):
    """A node in the artifact lineage graph: an artifact, a table, or a URL fetch.

    The descriptive fields (``principal``, ``build_env``, ``env_hash``, ``source``,
    ``content_sha256``) are best-effort: empty for tables, core transforms and
    older rows. ``principal`` is also empty for local notebook runs, which have
    no authenticated identity. For a ``fetch``, ``content_sha256`` is the digest
    of the bytes read and ``created_at``, when recorded, the time they were downloaded.
    """

    uri: str
    artifact_id: str | None = None
    version: int | None = None
    type: str  # "artifact" | "table" | "fetch"
    transform_ref: str | None = None
    created_at: float | None = None
    principal: str | None = None
    build_env: str = ""
    build_duration_ms: int = 0
    env_hash: str = ""
    source: str = ""
    content_sha256: str | None = None
    # A time-travel SQL cell's: the warehouse moment it read, and until when that
    # moment can be queried again (ISO-8601 UTC).
    snapshot_at: str | None = None
    snapshot_valid_until: str | None = None


class LineageEdge(BaseModel):
    """A lineage edge from an input to the artifact that used it, at ``input_version``."""

    from_uri: str
    to_uri: str
    input_version: str


class ArtifactLineageResponse(BaseModel):
    """An artifact's transitive input graph, with its direct inputs listed separately."""

    artifact_uri: str
    artifact_id: str
    version: int
    nodes: list[LineageNode]
    edges: list[LineageEdge]
    depth: int
    direct_inputs: list[str]


class DependentInfo(BaseModel):
    """An artifact that uses another as input, at ``input_version``."""

    artifact_uri: str
    artifact_id: str
    version: int
    name: str | None = None
    transform_ref: str | None = None
    created_at: float | None = None
    input_version: str


class ArtifactDependentsResponse(BaseModel):
    """Artifacts that use a given artifact as input (reverse dependencies)."""

    artifact_uri: str
    artifact_id: str
    version: int
    dependents: list[DependentInfo]
    total_count: int


# --- Executor Protocol v1: stable interface for external executors ---
# Push: Strata POSTs multipart Arrow IPC inputs to {executor_url}/v1/execute.
# Pull: the executor GETs /v1/builds/{build_id}/manifest, fetches inputs, uploads, finalizes.
EXECUTOR_PROTOCOL_VERSION = "v1"

EXECUTOR_PROTOCOL_HEADER = "X-Strata-Executor-Protocol"
EXECUTOR_LOGS_HEADER = "X-Strata-Logs"


class ExecutorInputDescriptor(BaseModel):
    """One input of an executor request; ``uri`` is informational only."""

    name: str
    format: str = "arrow_ipc_stream"
    uri: str | None = None
    byte_size: int | None = None


class ExecutorTransformSpec(BaseModel):
    """Transform sent to an executor: ref (e.g. ``duckdb_sql@v1``), code hash and params."""

    ref: str
    code_hash: str
    params: dict[str, object]


class ExecutorRequestMetadata(BaseModel):
    """The ``metadata`` JSON part of a push-model (v1) executor request."""

    protocol_version: str = EXECUTOR_PROTOCOL_VERSION
    build_id: str
    tenant: str | None = None
    principal: str | None = None
    provenance_hash: str
    transform: ExecutorTransformSpec
    inputs: list[ExecutorInputDescriptor]


class ExecutorResponse(BaseModel):
    """An executor's JSON error body; success returns an Arrow IPC stream instead."""

    success: bool
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None
    output_rows: int | None = None
    output_bytes: int | None = None
    logs: str | None = None


class ExecutorManifestInput(BaseModel):
    """One input of a pull-model manifest, fetched from a signed ``download_url``."""

    name: str
    download_url: str
    byte_size: int | None = None
    format: str = "arrow_ipc_stream"


class ExecutorManifest(BaseModel):
    """Pull-model build manifest.

    The executor downloads the inputs, runs the transform, uploads to
    ``upload_url`` within ``max_output_bytes``, then calls ``finalize_url``.
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
    """Capabilities an executor reports from ``GET /health``."""

    protocol_versions: list[str] = [EXECUTOR_PROTOCOL_VERSION]
    transform_refs: list[str] = []
    max_input_bytes: int | None = None
    max_output_bytes: int | None = None
    max_concurrent_executions: int | None = None
    features: dict[str, object] | None = None


class ExecutorHealthResponse(BaseModel):
    """An executor's ``GET /health`` body; ``status`` is healthy, degraded or unhealthy."""

    status: str  # "healthy" | "degraded" | "unhealthy"
    capabilities: ExecutorCapabilities
    version: str | None = None
    uptime_seconds: float | None = None
    active_executions: int | None = None
    # What the machine reports about itself; a field left out is unknown.
    hardware: dict[str, object] | None = None
