"""Pydantic models for notebook.toml and notebook state."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MountMode(StrEnum):
    """Access mode for a filesystem mount."""

    READ_ONLY = "ro"
    READ_WRITE = "rw"


class MountSpec(BaseModel):
    """A filesystem mount declaration.

    Mounts give cells transparent access to local and remote directories
    via standard ``pathlib.Path`` operations.  The mount ``name`` becomes
    a variable in the cell namespace bound to a local ``Path`` that the
    executor resolves before execution.

    Supported URI schemes: ``file://``, ``s3://``, ``gs://``, ``az://``.
    """

    name: str = Field(
        ...,
        description="Mount name — injected as a Path variable in the cell namespace",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    )
    uri: str = Field(
        ...,
        description="URI: file:///path, s3://bucket/prefix, gs://bucket/prefix, az://container/prefix",
    )
    mode: MountMode = Field(
        default=MountMode.READ_ONLY,
        description="Access mode: 'ro' (read-only) or 'rw' (read-write)",
    )
    pin: str | None = Field(
        default=None,
        description="Pinned version/etag — disables fingerprinting when set",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Backend storage options passed to fsspec (e.g. anon=True, "
            "endpoint_url, profile). Merged with executor-level credentials."
        ),
    )


class TableSpec(BaseModel):
    """An Iceberg table input declaration.

    Tables connect a cell to the lake with snapshot-level staleness: the
    table's current snapshot id is folded into the cell's provenance, so
    new data landing in the table makes the cell stale and the normal
    cascade machinery re-runs it. The executor injects two variables into
    the cell namespace: ``<name>`` (the table URI string) and
    ``<name>_snapshot`` (the resolved snapshot id) so the cell can scan
    deterministically at that snapshot.

    URI format: ``<warehouse>#<namespace>.<table>`` — e.g.
    ``file:///data/warehouse#nyc.trips`` or ``s3://bucket/wh#db.events``.
    """

    name: str = Field(
        ...,
        description="Variable name — injected as the table URI string; "
        "<name>_snapshot carries the resolved snapshot id",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    )
    uri: str = Field(
        ...,
        description="Table URI: <warehouse>#<namespace>.<table>",
    )
    snapshot_pin: int | None = Field(
        default=None,
        description="Pinned snapshot id — the cell never goes stale on new data when set",
    )


class ConnectionSpec(BaseModel):
    """A named database connection from ``[connections.<name>]``.

    SQL cells reference connections by name via ``# @sql connection=<name>``.
    Driver-specific top-level keys (``uri``, ``host``, ``account``,
    ``database``, ``role``, ``path``, ...) are preserved as-is; the
    ``DriverAdapter`` for the chosen ``driver`` interprets them.

    The ``auth`` block is intentionally separate so secret values (typed
    with ``${VAR}`` indirection) live in one well-known place; ``options``
    is for runtime tunables that don't change which objects the connection
    sees (e.g. ``application_name``, ``connect_timeout``).
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        ...,
        description=("Connection name — referenced by SQL cells via ``# @sql connection=<name>``."),
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    )
    driver: str = Field(
        ...,
        description="Driver name: ``postgresql``, ``sqlite``, ``snowflake``, ...",
    )
    auth: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Credential ``${VAR}`` indirections; values resolved at execute "
            "time. Never hashed into provenance."
        ),
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Driver-specific runtime options that don't change which objects the connection sees."
        ),
    )


class MalformedConnection(BaseModel):
    """A ``[connections.<name>]`` block that failed to parse.

    Preserved across notebook saves so a hand-edited mistake (typo,
    missing ``driver``, bad name pattern) doesn't get silently erased
    by an unrelated rewrite (cell add, worker change, etc.). The
    annotation_validation layer reads ``error`` to surface a
    user-visible diagnostic.
    """

    name: str = Field(..., description="Connection name as written in TOML")
    body: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw TOML body, preserved verbatim for round-trip",
    )
    error: str = Field(..., description="Reason this block failed validation")


class WorkerBackendType(StrEnum):
    """Execution backend type for notebook workers."""

    LOCAL = "local"
    EXECUTOR = "executor"


class WorkerConfig(BaseModel):
    """Backend-specific worker configuration.

    The known keys are typed for validation + discoverability; backend-specific
    extras pass through (``extra='allow'``) so a new backend can carry its own
    settings without a schema change. The ``executor`` backend uses ``url`` /
    ``transport`` / ``strata_url``; ``local`` carries none.
    """

    model_config = ConfigDict(extra="allow")

    url: str | None = None  # executor endpoint (executor backend)
    transport: str | None = None  # "direct" | … (executor backend)
    strata_url: str | None = None  # store URL the executor's cells point at
    token_env: str | None = None  # env var holding the executor's shared-secret token
    token: str | None = None  # literal executor token (dev only; don't commit)


class WorkerSpec(BaseModel):
    """A named worker declaration."""

    name: str = Field(
        ...,
        description="Worker name used in notebook metadata and cell overrides",
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$",
    )
    backend: WorkerBackendType = Field(
        default=WorkerBackendType.LOCAL,
        description="Worker backend type",
    )
    runtime_id: str | None = Field(
        default=None,
        description="Stable runtime fingerprint override for provenance",
    )
    config: WorkerConfig = Field(
        default_factory=WorkerConfig,
        description="Backend-specific worker configuration",
    )


class CellStatus(StrEnum):
    """Execution status of a cell."""

    IDLE = "idle"
    RUNNING = "running"
    READY = "ready"
    ERROR = "error"
    STALE = "stale"


class StalenessReason(StrEnum):
    """Reason a cell is stale (has invalidated cache)."""

    SELF = "self"  # Cell source code changed
    UPSTREAM = "upstream"  # Upstream artifact changed
    ENV = "env"  # Environment/lockfile changed
    FORCED = "forced"  # Forced re-run despite cache hit


class CellLanguage(StrEnum):
    """Source language of a notebook cell."""

    PYTHON = "python"
    PROMPT = "prompt"
    SQL = "sql"
    MARKDOWN = "markdown"
    # R is in flight via #53 — analyzer landed via #56, executor lands
    # via #57. The enum value exists so notebooks can declare R cells
    # before #57 ships; trying to execute one before #57 lands will
    # raise ``UnknownLanguageError`` from the executor registry, which
    # is the right shape (loud failure, not silent fallthrough to
    # Python).
    R = "r"
    # Interactive widget cells (P1: analyzer + DAG participation only). A
    # widget cell is declarative — it produces value artifacts from
    # user-set controls with no subprocess. The executor lands in P2; until
    # then, executing one raises ``UnknownLanguageError`` (loud, not silent).
    WIDGET = "widget"


class DiagnosticSeverity(StrEnum):
    """Severity of an annotation-validation diagnostic."""

    ERROR = "error"
    WARN = "warn"
    INFO = "info"


class ContentType(StrEnum):
    """Serialization format for cell outputs."""

    ARROW_IPC = "arrow/ipc"
    JSON = "json/object"
    IMAGE_PNG = "image/png"
    TEXT_MARKDOWN = "text/markdown"
    PICKLE = "pickle/object"
    ERROR = "error"


class AnnotationDiagnostic(BaseModel):
    """A validation finding for a cell's source annotations."""

    severity: DiagnosticSeverity = Field(..., description="Diagnostic severity")
    code: str = Field(..., description="Stable identifier, e.g. 'worker_unknown'")
    message: str = Field(..., description="Human-readable explanation")
    line: int | None = Field(default=None, description="1-based line in cell source")


class CellStaleness(BaseModel):
    """Staleness status for a cell."""

    status: CellStatus = Field(..., description="Status: ready, stale, idle, running, error")
    reasons: list[StalenessReason] = Field(
        default_factory=list, description="List of staleness reasons"
    )


class ArtifactInfo(BaseModel):
    """Lightweight artifact metadata for API responses."""

    id: str = Field(..., description="Artifact ID")
    version: int = Field(..., description="Version number")
    provenance_hash: str = Field(..., description="Provenance hash for deduplication")
    content_type: str = Field(
        ..., description="Content type (arrow/ipc, json/object, pickle/object)"
    )
    rows: int | None = Field(default=None, description="Number of rows (for tables)")
    bytes: int = Field(default=0, description="Size in bytes")
    created_at: float = Field(..., description="Creation timestamp")


class VariantGroupConfig(BaseModel):
    """Persisted active-variant pointer for one variant group.

    Group membership itself is declared by ``# @variant`` annotations in
    cell source; this entry only records which variant is currently active.
    Stored under ``[[variant_group]]`` in notebook.toml.
    """

    group: str = Field(
        ...,
        description="Variant group identifier (matches ``# @variant <group> <name>``)",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    )
    active: str = Field(
        ...,
        description=(
            "Active variant name within the group (ignored in sweep mode). "
            "Empty means 'first variant in source order' — the same fallback "
            "used when a group has no toml entry."
        ),
        pattern=r"^([a-zA-Z_][a-zA-Z0-9_]*)?$",
    )
    # No strict pattern: an unknown value must not crash notebook parsing.
    # Execution treats anything other than ``"sweep"`` as switch mode, and
    # ``annotation_validation`` surfaces a ``variant_mode_invalid`` diagnostic.
    mode: str = Field(
        "switch",
        description=(
            "Group execution mode: 'switch' (exactly one active variant) or "
            "'sweep' (all variants run; downstream consumes a {variant: value} "
            "dict). Unknown values are treated as 'switch'."
        ),
    )

    @property
    def is_sweep(self) -> bool:
        """Whether the group runs in sweep mode (fail-safe: only exact 'sweep')."""
        return self.mode == "sweep"


class VariantMember(BaseModel):
    """One member of a variant group, surfaced for frontend rendering."""

    cell_id: str = Field(..., description="Cell ID")
    name: str = Field(..., description="Variant name within the group")
    is_active: bool = Field(..., description="Whether this is the active variant")


class VariantGroupState(BaseModel):
    """Resolved variant-group state attached to NotebookState.

    ``members`` is in source order (matches the order the cells appear in
    ``notebook.toml``'s ``cells`` list); the frontend uses this for
    variant-tab ordering.
    """

    group: str = Field(..., description="Variant group identifier")
    active_name: str = Field(..., description="Active variant name")
    active_cell_id: str = Field(..., description="Active variant's cell ID")
    mode: str = Field(
        "switch",
        description=(
            "Group mode: 'switch' (one active member; tab clicks switch active) "
            "or 'sweep' (all members run; tab clicks are display-only and "
            "downstream consumes a {variant: value} dict)."
        ),
    )
    members: list[VariantMember] = Field(
        default_factory=list,
        description="All members of this group, in source order",
    )


class CellMeta(BaseModel):
    """Metadata for a single cell in notebook.toml."""

    id: str = Field(..., description="Unique cell ID (UUID-like)")
    file: str = Field(..., description="Path to cell source file (relative to cells/)")
    language: CellLanguage = Field(default=CellLanguage.PYTHON, description="Programming language")
    order: float = Field(default=0, description="Display order in notebook")
    worker: str | None = Field(
        default=None,
        description="Cell-level worker override (overrides notebook default)",
    )
    timeout: float | None = Field(
        default=None,
        description="Cell-level execution timeout override in seconds",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Cell-level environment variable overrides",
    )
    mounts: list[MountSpec] = Field(
        default_factory=list,
        description="Cell-level mount overrides (supplement/override notebook-level mounts)",
    )


class NotebookToml(BaseModel):
    """Notebook metadata from notebook.toml."""

    notebook_id: str = Field(..., description="Unique notebook ID")
    name: str = Field(default="Untitled Notebook", description="Human-readable name")
    owner: str | None = Field(
        default=None,
        description=(
            "Opaque identity string of the user who created this notebook. "
            "Stamped on create when STRATA_PERSONAL_MODE_USER_HEADER is set "
            "and the request carries that header. Unset (None) means "
            "'unowned' — visible/deletable by any caller. The string is "
            "intentionally opaque: it can be an email, a GitHub login, a "
            "service-mode principal, or any other stable identifier."
        ),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    cells: list[CellMeta] = Field(default_factory=list, description="Cell metadata")
    worker: str | None = Field(
        default=None,
        description="Notebook-level default worker name",
    )
    timeout: float | None = Field(
        default=None,
        description="Notebook-level default execution timeout in seconds",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Notebook-level default environment variables",
    )
    workers: list[WorkerSpec] = Field(
        default_factory=list,
        description="Registered workers (personal/dev mode)",
    )
    mounts: list[MountSpec] = Field(
        default_factory=list,
        description="Notebook-level filesystem mounts",
    )
    connections: list[ConnectionSpec] = Field(
        default_factory=list,
        description="Named database connections from ``[connections.<name>]``",
    )
    malformed_connections: list[MalformedConnection] = Field(
        default_factory=list,
        description=(
            "Connection blocks that failed to parse. Preserved verbatim "
            "across saves so users don't lose hand-edited config to a "
            "transient typo."
        ),
    )
    variant_groups: list[VariantGroupConfig] = Field(
        default_factory=list,
        description=(
            "Active-variant pointers for variant groups. Group membership "
            "is declared in cell source via ``# @variant``; only the "
            "active selection per group is committed here."
        ),
    )
    ai: dict[str, Any] = Field(
        default_factory=dict,
        description="Notebook-level LLM configuration persisted under [ai]",
    )
    secret_manager: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "External secret-manager config under [secret_manager]. "
            "Non-sensitive routing only (provider, project_id, environment, "
            "path); the token that authenticates lives in the process "
            "environment."
        ),
    )
    r: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "R environment metadata under ``[r]``. Mirrors the Python "
            "side's ``[environment]`` block (lockfile hash, last sync "
            "timestamp, R version). Populated by ``_renv_sync`` after "
            "a successful ``renv::restore()``."
        ),
    )
    # Preserved in TOML round-trip but not used at runtime
    artifacts: dict = Field(default_factory=dict)
    environment: dict = Field(default_factory=dict)
    cache: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CellOutput(BaseModel):
    """Output variable metadata from cell execution."""

    content_type: str = Field(
        ..., description="Type of content (arrow/ipc, json/object, pickle/object, error)"
    )
    rows: int | None = Field(default=None, description="Number of rows (for tables)")
    columns: list[str] | None = Field(default=None, description="Column names (for tables)")
    bytes: int = Field(default=0, description="Size in bytes")
    artifact_uri: str | None = Field(
        default=None,
        description="Artifact URI backing this display output",
    )
    preview: int | float | str | bool | list | dict | None = Field(
        default=None,
        description="Preview data (first 20 rows for tables, value for scalars)",
    )
    inline_data_url: str | None = Field(
        default=None,
        description="Inline data URL for display-only renderers like images",
    )
    markdown_text: str | None = Field(
        default=None,
        description="Markdown source for display-only markdown outputs",
    )
    width: int | None = Field(default=None, description="Display width in pixels")
    height: int | None = Field(default=None, description="Display height in pixels")
    error: str | None = Field(default=None, description="Error message if serialization failed")


class CellTestCase(BaseModel):
    """One pytest test case from a cell-test run."""

    name: str = Field(..., description="Test function name (pytest nodeid tail)")
    nodeid: str = Field(default="", description="Full pytest nodeid")
    outcome: str = Field(..., description="passed | failed | error | skipped")
    message: str = Field(
        default="",
        description="Failure/error message (the rewritten-assert diff for fails)",
    )


class CellTestResult(BaseModel):
    """Result of running a cell's unit tests, persisted in runtime state.

    Keyed by the ``(cell_source_hash, test_source_hash, input_fingerprint)``
    triple so the UI can mark the last result *stale* when the cell source,
    the test source, or the upstream inputs change since the run.
    """

    passed: int = Field(default=0, description="Number of passing tests")
    failed: int = Field(default=0, description="Number of failing assertions")
    errored: int = Field(default=0, description="Number of errored tests (setup/cell failures)")
    skipped: int = Field(default=0, description="Number of skipped tests")
    tests: list[CellTestCase] = Field(default_factory=list, description="Per-test outcomes")
    cell_source_hash: str = Field(default="", description="Cell source hash the run was against")
    test_source_hash: str = Field(default="", description="Test source hash the run was against")
    input_fingerprint: str = Field(default="", description="Upstream input hashes at run time")
    ran_at: int = Field(default=0, description="Run timestamp (epoch milliseconds)")
    pytest_unavailable: bool = Field(
        default=False,
        description="True when pytest is not importable in the notebook venv",
    )
    auto_installed: list[str] = Field(
        default_factory=list,
        description="Dev tools (e.g. pytest) auto-provisioned into the venv for this run",
    )


class CellState(BaseModel):
    """A cell with its source code loaded."""

    id: str = Field(..., description="Cell ID")
    source: str = Field(default="", description="Cell source code")
    test_source: str = Field(
        default="",
        description="pytest source for this cell's unit tests (cells/{id}.test.py)",
    )
    test_result: CellTestResult | None = Field(
        default=None,
        description="Last persisted cell-test result (None until tests are run)",
    )
    language: CellLanguage = Field(default=CellLanguage.PYTHON, description="Programming language")
    order: float = Field(default=0, description="Display order in notebook")
    status: CellStatus = Field(
        default=CellStatus.IDLE,
        description="Execution status",
    )
    defines: list[str] = Field(
        default_factory=list,
        description="Variable names defined by this cell",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Variable names referenced by this cell",
    )
    mutation_defines: list[str] = Field(
        default_factory=list,
        description=(
            "Subset of defines that came from in-place mutations "
            "(df['col'] = ...); the harness must serialize these "
            "even when id() is preserved across execution."
        ),
    )
    upstream_ids: list[str] = Field(
        default_factory=list, description="Cell IDs this cell depends on"
    )
    downstream_ids: list[str] = Field(
        default_factory=list, description="Cell IDs that depend on this cell"
    )
    worker: str | None = Field(
        default=None,
        description="Resolved persisted worker for this cell (notebook default + cell override)",
    )
    worker_override: str | None = Field(
        default=None,
        description="Persisted cell-level worker override from notebook.toml",
    )
    timeout: float | None = Field(
        default=None,
        description="Resolved persisted timeout for this cell in seconds",
    )
    timeout_override: float | None = Field(
        default=None,
        description="Persisted cell-level timeout override from notebook.toml",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Resolved persisted environment variables for this cell",
    )
    env_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Persisted cell-level environment overrides from notebook.toml",
    )
    mounts: list[MountSpec] = Field(
        default_factory=list,
        description="Resolved mounts for this cell (notebook-level + cell-level overrides)",
    )
    mount_overrides: list[MountSpec] = Field(
        default_factory=list,
        description="Persisted cell-level mount overrides from notebook.toml",
    )
    variant_group: str | None = Field(
        default=None,
        description=(
            "Variant group ID parsed from ``# @variant <group> <name>``. "
            "None for cells that aren't part of a group."
        ),
    )
    variant_name: str | None = Field(
        default=None,
        description="Variant name within ``variant_group``, parsed from source.",
    )
    variant_active: bool = Field(
        default=True,
        description=(
            "True for cells that aren't grouped or for the active member "
            "of a group. False for inactive variants — these are excluded "
            "from the producer map and consumed_variables."
        ),
    )
    annotation_diagnostics: list[AnnotationDiagnostic] = Field(
        default_factory=list,
        description="Validation findings for the cell's source annotations",
    )
    is_leaf: bool = Field(
        default=False,
        description="Whether this is a leaf node (no downstream consumers)",
    )
    staleness: CellStaleness | None = Field(default=None, description="Staleness status")
    artifact_uri: str | None = Field(
        default=None, description="URI of last stored artifact (legacy single-var)"
    )
    artifact_uris: dict[str, str] = Field(
        default_factory=dict,
        description="Per-variable artifact URIs: {var_name: uri}",
    )
    display_outputs: list[CellOutput] = Field(
        default_factory=list,
        description="Ordered persisted display outputs for the cell",
    )
    display_output: CellOutput | None = Field(
        default=None,
        description="Primary persisted display output for the cell (legacy last-item shim)",
    )
    console_stdout: str = Field(
        default="",
        description="Captured stdout from the last execution (persisted for reopen)",
    )
    console_stderr: str = Field(
        default="",
        description="Captured stderr from the last execution (persisted for reopen)",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether last execution was a cache hit",
    )
    execution_method: str | None = Field(
        default=None,
        description="Last execution method: cached, warm, cold, executor",
    )
    remote_worker: str | None = Field(
        default=None,
        description="Remote worker name used for the last remote execution",
    )
    remote_transport: str | None = Field(
        default=None,
        description="Remote transport used for the last remote execution",
    )
    remote_build_id: str | None = Field(
        default=None,
        description="Signed build id for the last remote execution, when applicable",
    )
    remote_build_state: str | None = Field(
        default=None,
        description="Last observed signed build state for remote execution metadata",
    )
    remote_error_code: str | None = Field(
        default=None,
        description="Structured remote execution error code for the last run, when available",
    )
    last_provenance_hash: str | None = Field(
        default=None,
        exclude=True,
        description="Runtime-only provenance hash from the last successful execution",
    )
    last_source_hash: str | None = Field(
        default=None,
        exclude=True,
        description="Runtime-only source hash from the last successful execution",
    )
    last_env_hash: str | None = Field(
        default=None,
        exclude=True,
        description="Runtime-only environment hash from the last successful execution",
    )
    widget_values: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description="Runtime-only current values of a widget cell's controls",
    )

    def serialize(self) -> dict[str, Any]:
        """Return the cell-only wire view of this cell.

        Combines ``model_dump()`` with the cell-derived overlays the
        frontend expects: flattened ``staleness_reasons``, the curated
        ``annotations`` payload from the source-comment block, and
        module-export classification for Python cells. Session-coupled
        overlays (display-output hydration, causality, shadow warnings)
        are added separately by ``NotebookSession.serialize_cell``.
        """
        # Local imports to avoid a cycle: annotations.py and
        # module_export.py both import from this module.
        from strata.notebook.annotations import parse_annotations
        from strata.notebook.module_export import build_module_export_plan

        data = self.model_dump()
        data["staleness_reasons"] = (
            [reason.value for reason in self.staleness.reasons]
            if self.staleness and self.staleness.reasons
            else []
        )
        data["annotations"] = parse_annotations(self.source).to_wire_payload()

        # Module-cell classification — drives the "module" pill in the
        # UI and the richer tooltip on the module_export_blocked
        # diagnostic. Only meaningful for Python cells; prompt and
        # markdown cells have no Python identifiers to export.
        if self.language == CellLanguage.PYTHON:
            export_plan = build_module_export_plan(self.source)
            has_code_export = any(
                symbol.kind in ("function", "async function", "class")
                for symbol in export_plan.exported_symbols.values()
            )
            # "Module cell" = pure source *and* actually exports code.
            # A lone ``x = 1`` is pure but it's not a module cell in
            # any useful sense — routing still takes the data path.
            data["is_module_cell"] = export_plan.is_exportable and has_code_export
            if data["is_module_cell"]:
                data["module_exports"] = [
                    {"name": name, "kind": symbol.kind}
                    for name, symbol in sorted(export_plan.exported_symbols.items())
                ]
        return data


class NotebookState(BaseModel):
    """Full notebook state for API responses."""

    id: str = Field(..., description="Notebook ID")
    name: str = Field(default="Untitled Notebook", description="Notebook name")
    owner: str | None = Field(
        default=None,
        description=(
            "Opaque identity of the notebook's owner, mirrored from "
            "notebook.toml. None for unowned notebooks."
        ),
    )
    worker: str | None = Field(
        default=None,
        description="Notebook-level default worker name",
    )
    timeout: float | None = Field(
        default=None,
        description="Notebook-level default execution timeout in seconds",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Notebook-level default environment variables",
    )
    env_sources: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-key provenance for env: 'manual' for values typed in the "
            "Runtime panel, or a provider name (e.g. 'infisical') for "
            "values fetched from a secret manager."
        ),
    )
    env_fetch_error: str | None = Field(
        default=None,
        description=(
            "Last secret-manager fetch error, if any. None on success or "
            "when no manager is configured."
        ),
    )
    env_fetched_at: str | None = Field(
        default=None,
        description="ISO-8601 timestamp of the last secret-manager fetch.",
    )
    workers: list[WorkerSpec] = Field(
        default_factory=list,
        description="Registered workers available to this notebook",
    )
    mounts: list[MountSpec] = Field(
        default_factory=list,
        description="Notebook-level filesystem mount defaults",
    )
    connections: list[ConnectionSpec] = Field(
        default_factory=list,
        description="Named database connections available to SQL cells",
    )
    malformed_connections: list[MalformedConnection] = Field(
        default_factory=list,
        description=(
            "Connection blocks that failed to parse, preserved so "
            "annotation_validation can surface them and the writer "
            "round-trip doesn't erase them."
        ),
    )
    secret_manager_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parsed [secret_manager] block from notebook.toml — provider / "
            "project_id / environment / path routing. Non-sensitive; the "
            "token that authenticates to the manager lives in the process "
            "environment, never here."
        ),
    )
    r: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "R environment metadata under ``[r]``: lockfile hash, last "
            "renv-sync timestamp, R version. Populated by ``_renv_sync`` "
            "after a successful ``renv::restore()``. Mirrors the Python "
            "side's ``[environment]`` block."
        ),
    )
    cells: list[CellState] = Field(default_factory=list, description="Cells with source")
    variant_groups: list[VariantGroupState] = Field(
        default_factory=list,
        description=(
            "Resolved variant groups, one entry per group declared in any "
            "cell's ``# @variant`` annotation. Frontend renders these as "
            "tabbed groups; inactive members are not part of the DAG."
        ),
    )
    variant_active_selections: dict[str, str] = Field(
        default_factory=dict,
        exclude=True,
        description=(
            "Raw {group: active_name} selections from notebook.toml's "
            "[[variant_group]] entries. Populated by the parser; consumed "
            "by session DAG build to resolve into ``variant_groups``."
        ),
    )
    variant_modes: dict[str, str] = Field(
        default_factory=dict,
        exclude=True,
        description=(
            "Raw {group: mode} ('switch' | 'sweep') from notebook.toml's "
            "[[variant_group]] entries. Populated by the parser; consumed by "
            "the DAG build (sweep → all variants run) and annotation validation."
        ),
    )
    path: Path | None = Field(
        default=None,
        exclude=True,
        description="Path to notebook directory (not serialized)",
    )
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)

    def get_cell(self, cell_id: str) -> CellState | None:
        """Return the cell with the given id, or None if not present.

        Single accessor used everywhere a cell needs to be looked up
        by id — routes/ws/executor/session/cascade previously inlined
        the same ``next(c for c in ... if c.id == cell_id)`` generator
        in 60+ places, which made the basic state-container access
        pattern invisible and drift-prone. Linear scan is fine: cell
        lists are typically dozens, not thousands, and this is a hot
        path only on per-keystroke DAG rebuilds where the lookup is
        already dwarfed by the analysis cost.
        """
        for cell in self.cells:
            if cell.id == cell_id:
                return cell
        return None

    model_config = ConfigDict(arbitrary_types_allowed=True)
