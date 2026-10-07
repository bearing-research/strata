"""Configuration for Strata with Pydantic validation and environment variable support."""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from strata.notebook.python_versions import (
    current_python_minor,
    discover_installed_python_minors,
    normalize_python_minor,
)
from strata.types import CacheGranularity

# --- ACL configuration types ---


logger = logging.getLogger(__name__)


class AclRule(BaseModel):
    """One ACL rule: matches when principal, tenant (if set) and any table glob all match.

    ``principal="*"`` and ``tenant=None`` match anyone; ``tables`` holds glob
    patterns such as ``"file:db.*"`` and must not be empty.
    """

    # ``extra="forbid"``: a dropped unknown key widens an access rule (``tenants = "acme"`` would
    # leave ``tenant`` None, matching every tenant) while the file still looks right.
    model_config = ConfigDict(frozen=True, extra="forbid")

    principal: str = "*"
    tenant: str | None = None
    tables: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reject_unmatchable_rule(self) -> AclRule:
        """Reject a rule with no table patterns, which could never match.

        An inert deny rule fails open, and defaulting to all tables would widen
        allow rules, so the operator must write ``tables = ["*"]`` explicitly.
        """
        if not self.tables:
            raise ValueError(
                "ACL rule must list at least one table pattern; a rule with no "
                'patterns can never match. Use tables = ["*"] to mean all tables.'
            )
        return self

    @field_validator("tables", mode="before")
    @classmethod
    def convert_tables_to_tuple(cls, v: Any) -> tuple[str, ...]:
        """Convert list to tuple for tables."""
        if isinstance(v, list):
            return tuple(v)
        return v


class AclConfig(BaseModel):
    """Access control list, evaluated deny rules first, then allow rules, then ``default``.

    ``deny`` / ``allow`` are accepted as aliases for the rule fields. Unknown keys
    are rejected: ignoring one would boot an ACL that enforces nothing.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    default: Literal["allow", "deny"] = "allow"
    deny_rules: list[AclRule] = Field(
        default_factory=list, validation_alias=AliasChoices("deny_rules", "deny")
    )
    allow_rules: list[AclRule] = Field(
        default_factory=list, validation_alias=AliasChoices("allow_rules", "allow")
    )


def _find_pyproject() -> Path | None:
    """Find pyproject.toml in current or parent directories."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def _load_from_pyproject() -> dict:
    """Load strata configuration from pyproject.toml [tool.strata] section."""
    pyproject_path = _find_pyproject()
    if pyproject_path is None:
        return {}

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    return data.get("tool", {}).get("strata", {})


def _parse_acl_config(raw: dict) -> AclConfig:
    """Parse a ``[tool.strata.acl]`` table, e.g.::

    default = "deny"
    deny = [{ principal = "*", tables = ["file:finance.*"] }]
    allow = [{ tenant = "data-platform", tables = ["file:analytics.*"] }]
    """
    return AclConfig.model_validate(raw)


# Personal mode's retention when nothing is configured (see
# StrataConfig.artifact_gc_interval_seconds): an hourly sweep, capped at the
# size of two row-group caches.
_PERSONAL_GC_INTERVAL_SECONDS = 3600.0
_PERSONAL_GC_MAX_BYTES = 20 * 1024 * 1024 * 1024


class StrataConfig(BaseSettings):
    """Configuration for Strata server and client.

    Precedence: defaults < pyproject.toml ``[tool.strata]`` < ``STRATA_*`` env vars
    < overrides passed to :meth:`load`.

    Examples
    --------
    ::

        [tool.strata]
        port = 8765
        cache_dir = "/tmp/strata-cache"
        s3_endpoint_url = "http://localhost:9000"

        [tool.strata.catalog_properties]
        type = "sql"
        uri = "sqlite:///catalog.db"
    """

    model_config = SettingsConfigDict(
        env_prefix="STRATA_",
        env_nested_delimiter="__",
        extra="ignore",  # Ignore extra fields from pyproject.toml
        # A field carrying validation_alias still has to be settable by its own
        # name, because load() passes pyproject keys as init kwargs.
        populate_by_name=True,
    )

    # Server settings
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8765

    # The origin readers reach this server on, when it differs from what the server sees (behind a
    # reverse proxy ``request.base_url`` is the internal address, which published pages and oEmbed
    # would advertise). Only publication URLs consult this; unset, the request's own origin is used.
    public_base_url: str | None = None
    # The path a reverse proxy serves this server under (``/o/acme/lab``), with or without the proxy
    # stripping it. Empty means the root. Every URL the server and the UI build carries it.
    public_base_path: str = ""

    # Cache settings
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".strata" / "cache")
    max_cache_size_bytes: Annotated[int, Field(gt=0)] = 10 * 1024 * 1024 * 1024  # 10 GB
    cache_granularity: CacheGranularity = CacheGranularity.ROW_GROUP_PROJECTION

    # Fetcher settings
    batch_size: Annotated[int, Field(gt=0)] = 65536  # rows per batch
    fetch_parallelism: Annotated[int, Field(ge=1)] = 4  # Max concurrent fetches per scan
    max_fetch_workers: Annotated[int, Field(ge=1)] = 32  # Max threads in fetch pool

    # Catalog settings (for pyiceberg)
    catalog_name: str = "default"
    catalog_properties: dict[str, str] = Field(default_factory=dict)
    # Named catalogs, each a set of pyiceberg catalog properties (``type`` =
    # ``rest``, ``glue``, ``sql``, ... plus that type's settings). A table in
    # one is addressed as ``<name>:<namespace>.<table>``, by ``@table`` and by
    # scans alike.
    catalogs: dict[str, dict[str, str]] = Field(default_factory=dict)

    # Resource limits (backpressure)
    max_concurrent_scans: Annotated[int, Field(ge=1)] = 100
    max_tasks_per_scan: Annotated[int, Field(ge=1)] = 1000
    plan_timeout_seconds: Annotated[float, Field(gt=0)] = 30.0
    scan_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0
    max_response_bytes: Annotated[int, Field(gt=0)] = 512 * 1024 * 1024  # 512 MB
    # Iceberg equality deletes a row group may need in memory at once (all
    # the delete rows whose key range can meet it). A scan over the limit is
    # refused while planning, with a pointer to compaction; see
    # iceberg_equality.
    max_equality_delete_rows: Annotated[int, Field(ge=0)] = 10_000_000
    # How long a finished or abandoned stream's state lingers before cleanup.
    stream_state_ttl_seconds: Annotated[float, Field(gt=0)] = 300.0

    # How other nodes reach this one, e.g. "https://strata-3.internal:8765". Unset means
    # single-node: the stream ownership table is never touched.
    #
    # Setting it asserts that several nodes run behind one address and that this URL reaches this
    # node. Streams cannot move between nodes (a live asyncio.Task and an in-memory ReadPlan are not
    # shareable), so a node asked for another's stream redirects to the owner instead of a 404 that
    # looks like "expired".
    node_advertised_url: str | None = None

    # QoS: Two-tier admission control
    interactive_slots: Annotated[int, Field(ge=1)] = 32
    bulk_slots: Annotated[int, Field(ge=1)] = 8
    interactive_max_bytes: Annotated[int, Field(gt=0)] = 10 * 1024 * 1024  # 10 MB
    interactive_max_columns: Annotated[int, Field(ge=1)] = 10
    interactive_queue_timeout: Annotated[float, Field(gt=0)] = 10.0
    bulk_queue_timeout: Annotated[float, Field(gt=0)] = 30.0
    per_client_interactive: Annotated[int, Field(ge=0)] = 2  # 0 disables per-client caps
    per_client_bulk: Annotated[int, Field(ge=0)] = 1

    metadata_db: Path | None = None

    # S3 settings
    s3_region: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_endpoint_url: str | None = None
    s3_anonymous: bool = False

    arrow_memory_pool: Literal["default", "system", "jemalloc", "mimalloc"] | None = None

    # Rate limiting settings
    rate_limit_enabled: bool = True
    rate_limit_global_rps: Annotated[float, Field(gt=0)] = 1000.0
    rate_limit_global_burst: Annotated[float, Field(gt=0)] = 100.0
    rate_limit_client_rps: Annotated[float, Field(gt=0)] = 100.0
    rate_limit_client_burst: Annotated[float, Field(gt=0)] = 20.0
    rate_limit_scan_rps: Annotated[float, Field(gt=0)] = 50.0
    rate_limit_warm_rps: Annotated[float, Field(gt=0)] = 10.0

    # S3 timeout settings
    s3_connect_timeout_seconds: Annotated[float, Field(gt=0)] = 10.0
    s3_request_timeout_seconds: Annotated[float, Field(gt=0)] = 30.0

    fetch_timeout_seconds: Annotated[float, Field(gt=0)] = 60.0

    # Adaptive concurrency control
    adaptive_enabled: bool = False
    adaptive_interval_seconds: Annotated[float, Field(gt=0)] = 5.0
    adaptive_target_p95_ms: Annotated[float, Field(gt=0)] = 500.0
    adaptive_min_interactive: Annotated[int, Field(ge=1)] = 4
    adaptive_max_interactive: Annotated[int, Field(ge=1)] = 64
    adaptive_min_bulk: Annotated[int, Field(ge=1)] = 2
    adaptive_max_bulk: Annotated[int, Field(ge=1)] = 32
    adaptive_hysteresis: Annotated[int, Field(ge=1)] = 3

    # Multi-tenancy settings
    multi_tenant_enabled: bool = False
    tenant_header: str = "X-Tenant-ID"
    require_tenant_header: bool = False
    # Per-tenant admission defaults come from interactive_slots / bulk_slots.

    # Trusted proxy authentication settings
    auth_mode: Literal["none", "trusted_proxy", "api_key"] = "none"
    proxy_token_header: str = "X-Strata-Proxy-Token"
    proxy_token: str | None = None
    principal_header: str = "X-Strata-Principal"
    scopes_header: str = "X-Strata-Scopes"
    hide_forbidden_as_not_found: bool = True

    # Opt-in: let authenticated clients WRITE in service mode (put / set_name / set_alias / tags),
    # scoped to the caller's tenant and gated by the `artifacts:write` scope. Off, service mode is
    # read-only. Requires trusted-proxy auth so writes are attributable.
    service_writes_enabled: bool = False

    acl_config: AclConfig = Field(default_factory=AclConfig)

    # Default ``personal``: single-user, loopback-only, works out of the box. Multi-user and
    # multi-tenant deployments opt in with ``"service"`` plus matching auth / artifact settings,
    # checked by ``validate_mode_coherence``.
    deployment_mode: Literal["service", "personal"] = "personal"
    allow_remote_clients_in_personal: bool = False
    # Extra browser origins allowed to make cross-origin calls. Same-origin is always allowed; this
    # exists for `npm run dev` (Vite on another port, via VITE_STRATA_URL), e.g.
    # ["http://localhost:5173"].
    #
    # Empty on purpose: a permissive origin lets any page the user visits drive the loopback API,
    # which in personal mode has no auth and can execute arbitrary Python. Loopback binding is no
    # defence; the browser runs there too.
    cors_allow_origins: list[str] = []
    # Host header names this server answers to, beyond loopback names, IP literals and ``host``.
    # Without the check a DNS-rebinding page shares an origin with the server and passes the
    # origin guard. Always on in personal mode; service mode checks only when this is set. Exact
    # names or a leading dot for a suffix (``.example.com``); ``*`` turns the check off.
    allowed_hosts: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Mount the MCP server at ``/mcp`` so an external coding agent can drive the live session over
    # streamable HTTP. It exposes the read/run/author surface, so ``validate_mode_coherence``
    # allows it in personal mode, or in service mode only with principal auth, where each tool
    # call is checked against its caller's scopes. Needs the ``[mcp]`` extra; without it the flag
    # warns and no-ops.
    mcp_enabled: bool = False
    # Origins allowed to embed a notebook's app view in an ``<iframe>``; sets
    # ``Content-Security-Policy: frame-ancestors 'self' <origins>``. Empty (the default) means
    # same-origin only. A JSON array or comma-separated list of origins
    # (``https://analytics.example.com``) or ``*`` for any host.
    embed_frame_ancestors: Annotated[list[str], NoDecode] = Field(default_factory=list)
    artifact_dir: Path | None = None
    # Builds stuck in 'building' longer than this are demoted to failed at startup: they can never
    # serve data and would otherwise linger forever.
    artifact_zombie_build_timeout_seconds: Annotated[float, Field(gt=0)] = 3600.0
    # Retention for the server's artifact store (ArtifactStore.garbage_collect). Every
    # artifact_gc_interval_seconds, collect what nothing holds (no name, alias, pin or publication,
    # and not the current value of a chosen id), least recently used first: anything idle past
    # artifact_gc_max_idle_days, plus enough to bring a store over artifact_gc_max_bytes down to 80%
    # of it, but never anything used in the last artifact_gc_min_idle_seconds.
    #
    # Unset, personal mode sweeps hourly with a 20 GiB cap (a laptop store otherwise grows forever);
    # service mode sweeps only when the operator sets the interval, since a team store's retention
    # is their call. 0 turns off the interval, the cap or the idle limit.
    artifact_gc_interval_seconds: Annotated[float, Field(ge=0)] | None = None
    artifact_gc_max_bytes: Annotated[int, Field(ge=0)] | None = None
    artifact_gc_max_idle_days: Annotated[float, Field(ge=0)] = 30.0
    artifact_gc_min_idle_seconds: Annotated[float, Field(ge=0)] = 3600.0
    # A notebook's own store (<notebook>/.strata/artifacts) keeps each cell
    # output's current value plus this many earlier ones, so reverting a
    # recent edit is still a cache hit; older values are pruned when the
    # server opens the notebook. 0 turns pruning off and keeps every value.
    notebook_keep_superseded_versions: Annotated[int, Field(ge=0)] = 3
    # Registry aliases that require approval: moves/deletes of these aliases
    # (e.g. "champion") land in a pending queue instead of applying, and an
    # explicit approve applies them. Empty (the default) = no gating.
    registry_protected_aliases: Annotated[list[str], NoDecode] = Field(default_factory=list)
    notebook_storage_dir: Path = Field(
        default_factory=lambda: Path.home() / ".strata" / "notebooks"
    )
    notebook_python_versions: Annotated[list[str], NoDecode] = Field(
        default_factory=discover_installed_python_minors
    )
    # How a notebook's Python environment is kept: "uv" gives each notebook its
    # own .venv; "shared" links notebooks with the same lockfile and
    # interpreter to one environment under notebook_shared_env_dir (default:
    # "envs" beside notebook_storage_dir). Shared environments nothing links to
    # are removed once unused for notebook_shared_env_ttl_days. POSIX only.
    notebook_env_backend: Literal["uv", "shared"] = "uv"
    notebook_shared_env_dir: Path | None = None
    notebook_shared_env_ttl_days: Annotated[float, Field(ge=0)] = 7.0

    # Point the ambient `strata` client in notebook cells at a REMOTE shared store instead of this
    # server. `notebook_remote_store_headers` carries the auth it needs (proxy identity/token or a
    # bearer token); set it via env so secrets stay out of committed config.
    notebook_remote_store_url: str | None = None
    notebook_remote_store_headers: dict[str, str] = Field(default_factory=dict)
    # Send the caller's principal to the remote store as X-Strata-Principal,
    # replacing any the static headers name, so a shared server's results,
    # promotions and approvals are attributed to the member and not the server.
    # Off for a remote store that expects one fixed service identity.
    notebook_remote_store_forward_principal: bool = True
    # Consult the remote store on a LOCAL cache miss, so a colleague's expensive cell becomes your
    # instant result. Unlike the URL above (explicit publish), this covers unnamed intermediates,
    # which is where the recomputation is.
    #
    # Opt-in and separate from the URL: it puts bytes another machine produced into your store and
    # adds a round-trip to every cell's miss path.
    notebook_team_cache_enabled: bool = False

    # What the cache offers outward. On a personal server, offering everything puts every
    # intermediate a researcher computed into the team's store.
    #
    # - all: every downstream-consumed variable of every successful cell; right for a server whose
    #   purpose is a shared cache.
    # - promoted: offer nothing automatically; `strata artifact promote` shares a result. Pulls are
    #   unchanged.
    # - off: no offers and no pulls, keeping the URL a cell's ambient client still needs.
    notebook_team_cache_publish: Literal["all", "promoted", "off"] = "all"

    # Which server env vars a cell subprocess gets. Empty (the default) passes everything: right on
    # a laptop, but on a shared server any cell can read the remote-store headers, proxy token,
    # worker tokens and every data-source credential.
    #
    # Exact names, or a prefix with a trailing ``*``. Essentials a subprocess needs are always
    # included; STRATA_* is dropped unless named exactly. A cell's own ``[env]`` and mount
    # credentials travel in the manifest, not the environment.
    notebook_harness_env_allowlist: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # The OS user a cell subprocess runs as. The allowlist above filters what a cell gets; this
    # changes who it is, so it cannot read /proc/<server pid>/environ or the server's files.
    #
    # Service mode refuses to run cell code on its own host unless this is set (the alternative is a
    # managed worker on another machine). The server must run as root to switch users. POSIX only.
    notebook_harness_user: str | None = None

    # Named credentials, referenced from notebook.toml by name so no secret is
    # committed: ``{name: {field: value}}``, where each value is usually a
    # ``${VAR}`` resolved against the notebook's environment (which a secret
    # manager fills) and then the server's. A mount's fields become fsspec
    # storage options; a connection's become driver auth.
    notebook_credentials: Annotated[dict[str, dict[str, str]], NoDecode] = Field(
        default_factory=dict
    )
    # A default credential per mount URI scheme (``{"s3": "org-bucket"}``), for
    # mounts that name none, so the organization's primary store works without
    # any notebook change.
    notebook_mount_credentials: Annotated[dict[str, str], NoDecode] = Field(default_factory=dict)

    # Hosts ``@fetch``, and a prompt cell's ``[ai] base_url`` from
    # notebook.toml, may reach even on a private address, e.g. an internal data
    # or model server. Public hosts need no entry; private, loopback and
    # link-local addresses are refused unless named here. Exact names, or a leading dot
    # for a suffix (``.internal``). Same rule as STRATA_WORKER_ALLOWED_HOSTS.
    notebook_fetch_allowed_hosts: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # How long the last person to change a notebook cell holds it: an edit by
    # someone else inside the window is refused with ``cell_locked`` unless it
    # is forced. 0 turns the soft lock off.
    notebook_cell_lock_seconds: Annotated[float, Field(ge=0)] = 5.0

    # Open notebook sessions. Each open notebook keeps this many pre-spawned
    # Python (and R) processes; 0 turns the pools off. A session nobody has
    # edited, run or focused for the TTL is closed, with a tab connected or
    # not, and beyond the maximum the least recently used are closed. With the
    # memory floor set (Linux), idle sessions are closed while the host's
    # available memory is below it. Closing loses only the warm processes.
    notebook_warm_pool_size: Annotated[int, Field(ge=0)] = 2
    notebook_session_ttl_seconds: Annotated[float, Field(gt=0)] = 4 * 3600.0
    notebook_max_sessions: Annotated[int, Field(gt=0)] = 50
    notebook_session_min_available_mb: Annotated[int, Field(gt=0)] | None = None

    # LLM settings for prompt cells (OpenAI-compatible API)
    ai_base_url: str | None = None
    ai_model: str | None = None
    ai_api_key: str | None = None
    ai_max_output_tokens: Annotated[int, Field(gt=0)] = 4096
    ai_timeout_seconds: Annotated[float, Field(gt=0)] = 60.0

    # Metadata backend for the artifact store. Unset means SQLite in artifact_dir. A DSN moves the
    # system of record to a shared server so several nodes can share one store. Blobs are configured
    # separately (artifact_blob_backend); a shared database with local blobs only works on one
    # machine, so the two are validated together below.
    artifact_metadata_dsn: str | None = None

    artifact_blob_backend: Literal["local", "s3", "gcs", "azure"] = "local"
    artifact_s3_bucket: str | None = None
    artifact_s3_prefix: str = "artifacts"
    artifact_gcs_bucket: str | None = None
    artifact_gcs_prefix: str = "artifacts"
    artifact_azure_container: str | None = None
    artifact_azure_prefix: str = "artifacts"

    # GCS configuration
    #
    # PyArrow's GcsFileSystem has no project parameter: the legacy STRATA_GCS_PROJECT_ID always fed
    # ``default_bucket_location`` (a location like ``US``), so it stays accepted with that
    # behaviour; ``validate_gcs_settings`` says what it controls.
    #
    # Both aliases carry the STRATA_ prefix: validation_alias replaces env_prefix, so a bare
    # "gcs_project_id" would make the ambient GCP variable GCS_PROJECT_ID live config.
    gcs_default_bucket_location: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "STRATA_GCS_DEFAULT_BUCKET_LOCATION",
            "STRATA_GCS_PROJECT_ID",
        ),
    )
    # Either a path to a service-account key file or the key material itself.
    # The name invites pasting JSON, which container deployments want to do,
    # so both are accepted (see ``GCSBlobStore``).
    gcs_credentials_json: str | None = None
    gcs_anonymous: bool = False
    gcs_endpoint_override: str | None = None

    # Azure Blob Storage configuration
    azure_account_name: str | None = None
    azure_account_key: str | None = None
    azure_connection_string: str | None = None
    azure_sas_token: str | None = None
    azure_use_default_credential: bool = False
    azure_endpoint_url: str | None = None  # For Azurite emulator

    # Server-mode transforms configuration
    transforms_config: dict = Field(default_factory=dict)

    # Transform execution mode:
    # - "embedded": Use embedded executor for local deployment (default)
    #   Common transforms like duckdb_sql@v1 run in-process, no external service needed.
    # - "registry": Only use transforms explicitly configured in transforms_config.
    #   Requires external executor services for all transforms.
    transform_mode: Literal["embedded", "registry"] = "embedded"

    # Build runner configuration
    build_runner_poll_interval_ms: Annotated[int, Field(ge=1)] = 500
    build_runner_max_concurrent: Annotated[int, Field(ge=1)] = 10
    build_runner_max_per_tenant: Annotated[int, Field(ge=1)] = 3
    build_runner_default_timeout: Annotated[float, Field(gt=0)] = 300.0
    build_runner_default_max_output: Annotated[int, Field(gt=0)] = 1024 * 1024 * 1024  # 1 GB

    # Pull model configuration
    signed_url_expiry_seconds: Annotated[float, Field(gt=0)] = 600.0
    # Put object-store URLs in build manifests where the blob store can sign
    # them (S3, GCS, Azure), so a worker's inputs and output bypass this
    # server. Off by default: the output then arrives as a form upload or an
    # Azure PUT, which a worker older than this does not send.
    artifact_presigned_urls: bool = False
    # How long a remote dispatch may wait for its job to start running when the
    # worker (or a pool in front of it) answers 202 with a job to poll. Separate
    # from the cell's own timeout, which starts only once the job is running, so
    # a cold machine does not spend the cell's budget booting.
    worker_provisioning_timeout_seconds: Annotated[float, Field(gt=0)] = 600.0
    # HMAC secret for signing pull-model build URLs. Unset means a random per-process secret, so
    # signed URLs break on restart and differ across replicas. Set a stable value for any
    # multi-replica or restart-surviving deployment.
    transform_signing_secret: str | None = None

    # Build QoS configuration
    build_qos_interactive_slots: Annotated[int, Field(ge=1)] = 16
    build_qos_bulk_slots: Annotated[int, Field(ge=1)] = 8
    build_qos_per_tenant_interactive: Annotated[int, Field(ge=1)] = 4
    build_qos_per_tenant_bulk: Annotated[int, Field(ge=1)] = 2
    build_qos_interactive_timeout: Annotated[float, Field(gt=0)] = 5.0
    build_qos_bulk_timeout: Annotated[float, Field(gt=0)] = 15.0
    build_qos_per_tenant_timeout: Annotated[float, Field(gt=0)] = 1.0
    build_qos_bytes_per_day: int | None = None
    build_qos_bulk_bytes_threshold: Annotated[int, Field(gt=0)] = 100 * 1024 * 1024  # 100MB
    build_qos_bulk_inputs_threshold: Annotated[int, Field(ge=1)] = 5

    @field_validator(
        "cache_dir",
        "metadata_db",
        "artifact_dir",
        "notebook_storage_dir",
        mode="before",
    )
    @classmethod
    def convert_str_to_path(cls, v: Any) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("cache_granularity", mode="before")
    @classmethod
    def convert_cache_granularity(cls, v: Any) -> CacheGranularity:
        """Convert string to CacheGranularity enum."""
        if isinstance(v, str):
            return CacheGranularity(v)
        return v

    @field_validator("registry_protected_aliases", mode="before")
    @classmethod
    def normalize_registry_protected_aliases(cls, v: Any) -> list[str]:
        """Accept list, JSON array, or comma-separated alias names."""
        if v is None:
            return []
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                import json

                parsed = json.loads(stripped)
                if not isinstance(parsed, list):
                    raise ValueError("registry_protected_aliases must be a list")
                v = parsed
            else:
                v = [part.strip() for part in stripped.split(",") if part.strip()]
        if not isinstance(v, list):
            raise ValueError("registry_protected_aliases must be a list")
        return [str(item) for item in v]

    @field_validator("notebook_harness_env_allowlist", mode="before")
    @classmethod
    def normalize_harness_env_allowlist(cls, v: Any) -> list[str]:
        """Accept list, JSON array, or comma-separated variable names."""
        if v is None:
            return []
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                import json

                parsed = json.loads(stripped)
                if not isinstance(parsed, list):
                    raise ValueError("notebook_harness_env_allowlist must be a list")
                v = parsed
            else:
                v = [part.strip() for part in stripped.split(",") if part.strip()]
        if not isinstance(v, list):
            raise ValueError("notebook_harness_env_allowlist must be a list")
        return [str(item) for item in v]

    @field_validator("notebook_credentials", "notebook_mount_credentials", mode="before")
    @classmethod
    def parse_credential_maps(cls, v: Any, info: Any) -> dict:
        """Accept a dict or a JSON object string (the env-var form)."""
        if v is None or v == "":
            return {}
        if isinstance(v, str):
            import json

            v = json.loads(v)
        if not isinstance(v, dict):
            raise ValueError(f"{info.field_name} must be a JSON object")
        return v

    @field_validator("notebook_fetch_allowed_hosts", "allowed_hosts", mode="before")
    @classmethod
    def normalize_fetch_allowed_hosts(cls, v: Any) -> list[str]:
        """Accept a list or comma-separated host names."""
        if v is None:
            return []
        if isinstance(v, str):
            v = [part.strip() for part in v.split(",") if part.strip()]
        return [str(item).lower() for item in v]

    @field_validator("public_base_path", mode="before")
    @classmethod
    def normalize_public_base_path(cls, v: Any) -> str:
        """Accept ``/a/b``, ``a/b/`` or ``/`` alike; store ``/a/b``, or ``""`` for the root."""
        path = str(v or "").strip().strip("/")
        if any(char in path for char in "?#%\\") or any(char.isspace() for char in path):
            raise ValueError(f"public_base_path must be a plain URL path, got {v!r}")
        return f"/{path}" if path else ""

    @field_validator("embed_frame_ancestors", mode="before")
    @classmethod
    def normalize_embed_frame_ancestors(cls, v: Any) -> list[str]:
        """Accept list, JSON array, or comma-separated origins."""
        if v is None:
            return []
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                import json

                parsed = json.loads(stripped)
                if not isinstance(parsed, list):
                    raise ValueError("embed_frame_ancestors must be a list")
                v = parsed
            else:
                v = [part.strip() for part in stripped.split(",") if part.strip()]
        if not isinstance(v, list):
            raise ValueError("embed_frame_ancestors must be a list")
        return [str(item) for item in v]

    @field_validator("notebook_python_versions", mode="before")
    @classmethod
    def normalize_notebook_python_versions(cls, v: Any) -> list[str]:
        """Accept list, JSON array, or comma-separated notebook Python versions."""
        if v is None:
            return [current_python_minor()]
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return [current_python_minor()]
            if stripped.startswith("["):
                import json

                parsed = json.loads(stripped)
                if not isinstance(parsed, list):
                    raise ValueError("notebook_python_versions must be a list")
                v = parsed
            else:
                v = [part.strip() for part in stripped.split(",") if part.strip()]

        if not isinstance(v, list):
            raise ValueError("notebook_python_versions must be a list")

        normalized: list[str] = []
        seen: set[str] = set()
        for item in v:
            if not isinstance(item, str):
                raise ValueError("notebook_python_versions entries must be strings")
            python_version = normalize_python_minor(item)
            if python_version not in seen:
                normalized.append(python_version)
                seen.add(python_version)
        if not normalized:
            raise ValueError("notebook_python_versions must not be empty")
        return normalized

    @model_validator(mode="after")
    def setup_paths_and_defaults(self) -> StrataConfig:
        """Set up paths and defaults after model creation."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.metadata_db is None:
            self.metadata_db = Path.home() / ".strata" / "meta.sqlite"
        if self.metadata_db is not None:
            self.metadata_db.parent.mkdir(parents=True, exist_ok=True)

        if self.artifact_dir is None and self.deployment_mode == "personal":
            self.artifact_dir = Path.home() / ".strata" / "artifacts"

        # Personal mode keeps its artifact store bounded unless told not to
        # (see artifact_gc_interval_seconds).
        if self.deployment_mode == "personal":
            if self.artifact_gc_interval_seconds is None:
                self.artifact_gc_interval_seconds = _PERSONAL_GC_INTERVAL_SECONDS
            if self.artifact_gc_max_bytes is None:
                self.artifact_gc_max_bytes = _PERSONAL_GC_MAX_BYTES

        if self.deployment_mode == "personal" and self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.notebook_storage_dir.mkdir(parents=True, exist_ok=True)

        return self

    @model_validator(mode="after")
    def validate_adaptive_ranges(self) -> StrataConfig:
        """Validate the adaptive controller's bounds and its starting point.

        The controller starts from the configured slot counts, so a count outside
        ``[min, max]`` would jump to a bound on the first adjustment.
        """
        if not self.adaptive_enabled:
            return self
        if self.adaptive_min_interactive > self.adaptive_max_interactive:
            raise ValueError(
                f"adaptive_min_interactive ({self.adaptive_min_interactive}) "
                f"cannot exceed adaptive_max_interactive ({self.adaptive_max_interactive})"
            )
        if self.adaptive_min_bulk > self.adaptive_max_bulk:
            raise ValueError(
                f"adaptive_min_bulk ({self.adaptive_min_bulk}) "
                f"cannot exceed adaptive_max_bulk ({self.adaptive_max_bulk})"
            )
        if not (
            self.adaptive_min_interactive <= self.interactive_slots <= self.adaptive_max_interactive
        ):
            raise ValueError(
                f"interactive_slots ({self.interactive_slots}) is outside the adaptive range "
                f"[{self.adaptive_min_interactive}, {self.adaptive_max_interactive}]; the "
                "controller starts from interactive_slots, so it would jump to a bound on its "
                "first adjustment"
            )
        if not (self.adaptive_min_bulk <= self.bulk_slots <= self.adaptive_max_bulk):
            raise ValueError(
                f"bulk_slots ({self.bulk_slots}) is outside the adaptive range "
                f"[{self.adaptive_min_bulk}, {self.adaptive_max_bulk}]; the controller starts "
                "from bulk_slots, so it would jump to a bound on its first adjustment"
            )
        if self.multi_tenant_enabled:
            raise ValueError(
                "adaptive_enabled cannot be combined with multi_tenant_enabled: the controller "
                "steers one tenant's limiters (the default tenant's), which in a multi-tenant "
                "deployment is a tier no request acquires"
            )
        return self

    @model_validator(mode="after")
    def validate_team_cache(self) -> StrataConfig:
        """Reject the team cache without ``notebook_remote_store_url``, in any mode.

        Without a store it is silently inert: every cell would recompute.
        """
        if self.notebook_team_cache_enabled and not self.notebook_remote_store_url:
            raise ValueError(
                "notebook_team_cache_enabled=True without notebook_remote_store_url "
                "(there is no store to look results up in, so every lookup would "
                "miss and every cell would recompute; set notebook_remote_store_url)"
            )
        return self

    @model_validator(mode="after")
    def validate_catalog_uris(self) -> StrataConfig:
        """Reject an object-store warehouse whose SQL catalog has no ``uri``.

        Without one the catalog is SQLite on this server's disk, invisible to
        every other reader of the bucket.
        """
        configured = {"catalog_properties": self.catalog_properties}
        configured.update({f"catalogs.{name}": props for name, props in self.catalogs.items()})
        for where, props in configured.items():
            warehouse = props.get("warehouse", "")
            credential = self.notebook_credentials.get(props.get("credential", ""), {})
            if (
                "://" in warehouse
                and not warehouse.startswith("file://")
                and props.get("type", "sql") == "sql"
                and "uri" not in props
                and "uri" not in credential
            ):
                raise ValueError(
                    f"{where} has the object-store warehouse {warehouse} but no "
                    "catalog uri, so its tables would live in a SQLite catalog on "
                    "this server's disk. Set the catalog database's uri (for "
                    "catalog_properties, STRATA_CATALOG_URI), e.g. "
                    "postgresql://user:pass@host/iceberg_catalog."
                )
        return self

    @model_validator(mode="after")
    def reject_a_remote_store_that_is_this_server(self) -> StrataConfig:
        """Reject a ``notebook_remote_store_url`` naming this server.

        Registry routes forward to that URL, so they would recurse until they time out.
        """
        url = (self.notebook_remote_store_url or "").strip().rstrip("/")
        if url and url == self.server_url:
            raise ValueError(
                f"notebook_remote_store_url is this server ({url}); a remote "
                "store has to be a different one, or the registry routes "
                "forward to themselves. Unset it for a single-machine setup."
            )
        return self

    @model_validator(mode="after")
    def warn_on_gcs_project_id(self) -> StrataConfig:
        """Warn that ``STRATA_GCS_PROJECT_ID`` sets the default bucket location, not a project.

        PyArrow's GcsFileSystem takes no project, so a project id there is a bogus
        location.
        """
        import os

        # Only when the old name is what supplied the value: with both set the
        # new one wins and the old is already being ignored, so saying "rename
        # it" would be advice about a setting that is doing nothing.
        if os.environ.get("STRATA_GCS_PROJECT_ID") and not os.environ.get(
            "STRATA_GCS_DEFAULT_BUCKET_LOCATION"
        ):
            logger.warning(
                "STRATA_GCS_PROJECT_ID does not set a GCP project: GcsFileSystem "
                "has no project parameter. Its value is used as the default bucket "
                "location (a GCS location such as 'US' or 'europe-west1'). Rename "
                "it to STRATA_GCS_DEFAULT_BUCKET_LOCATION, and check the value is "
                "a location rather than a project id."
            )
        return self

    @model_validator(mode="after")
    def validate_mode_coherence(self) -> StrataConfig:
        """Reject deployment-mode combinations that indicate misconfiguration.

        Personal mode is one identity with no tenants or proxy, so auth or
        multi-tenancy there means service-mode flags were copied by mistake.
        Service mode rejects settings whose security or build intent would be inert.
        """
        # Service mode: reject configs whose security/build intent is silently
        # inert.
        if self.deployment_mode == "service":
            conflicts: list[str] = []

            # Multi-tenancy is an access-control boundary: without auth the tenant header is
            # spoofable and direct artifact reads aren't tenant-filtered.
            if self.multi_tenant_enabled and not self.principal_auth_enabled:
                conflicts.append(
                    f"multi_tenant_enabled=True with auth_mode={self.auth_mode!r} "
                    "(the tenant header is unauthenticated and spoofable, and "
                    "reads aren't tenant-filtered without auth; set "
                    "auth_mode='trusted_proxy' or 'api_key')"
                )

            # Trusted-proxy auth without a shared token is no auth: verify_proxy_token() returns
            # True when none is configured, so any client could spoof the principal/scope headers.
            if self.auth_mode == "trusted_proxy" and not self.proxy_token:
                conflicts.append(
                    "auth_mode='trusted_proxy' without proxy_token (the token is "
                    "unset, so every request is accepted and principal/scope "
                    "headers can be spoofed; set proxy_token)"
                )

            # Writes are stamped with the caller's principal/tenant, which only exist under auth.
            # Trusted-proxy only: stamping identity into stored artifacts is a wider claim than a
            # read gate. The message names the mode actually seen.
            if self.service_writes_enabled and self.auth_mode != "trusted_proxy":
                conflicts.append(
                    f"service_writes_enabled=True with auth_mode={self.auth_mode!r} "
                    "(writes are attributed to and scoped by the caller's "
                    "principal/tenant, which require trusted-proxy auth; set "
                    "auth_mode='trusted_proxy')"
                )

            # Key auth has nowhere to keep keys without an artifact directory:
            # the api_keys table lives in the artifact store's database. Every
            # request would then fail closed at the middleware, which is safe
            # but useless -- reject at startup where it is diagnosable.
            if self.auth_mode == "api_key" and self.artifact_dir is None:
                conflicts.append(
                    "auth_mode='api_key' without artifact_dir (API keys are stored "
                    "in the artifact store's database; set artifact_dir)"
                )

            # A shared metadata store with local blobs only works on one machine: node B resolves
            # metadata from the shared database, then looks for bytes on node A's disk, failing long
            # after the write looked successful. Moving metadata off SQLite only serves multiple
            # nodes, so this is always a misconfiguration in service mode.
            if self.artifact_metadata_dsn and self.artifact_blob_backend == "local":
                conflicts.append(
                    "artifact_metadata_dsn with artifact_blob_backend='local' "
                    "(the metadata is shared across nodes but the blobs are not, "
                    "so another node resolves an artifact and then cannot read "
                    "its bytes; set artifact_blob_backend to s3, gcs, or azure)"
                )

            # The artifact store is created only when artifact_dir is set, even when its metadata
            # and blobs live elsewhere; without it every artifact route answers 404 or 500.
            artifact_store_configured = (
                self.artifact_metadata_dsn is not None
                or self.artifact_blob_backend != "local"
                or self.service_writes_enabled
            )
            if artifact_store_configured and self.artifact_dir is None:
                conflicts.append(
                    "an artifact store (artifact_metadata_dsn, a non-local "
                    "artifact_blob_backend or service_writes_enabled) without "
                    "artifact_dir (the store is only created when artifact_dir is "
                    "set, so every artifact route would fail; set artifact_dir to "
                    "a node-local directory, which holds nothing durable when the "
                    "metadata and blobs are shared)"
                )

            # ACL rules are only evaluated for an authenticated principal; without auth, configured
            # rules would be silently ignored.
            acl_configured = (
                self.acl_config.default != "allow"
                or bool(self.acl_config.deny_rules)
                or bool(self.acl_config.allow_rules)
            )
            if acl_configured and not self.principal_auth_enabled:
                conflicts.append(
                    f"acl_config rules with auth_mode={self.auth_mode!r} (ACL is "
                    "only enforced when the caller is authenticated; set "
                    "auth_mode='trusted_proxy' or 'api_key')"
                )

            # Transform builds persist artifacts, which need an artifact store (its metadata DB
            # lives under artifact_dir). Reject at startup rather than fail every build.
            if self.server_transforms_enabled and self.artifact_dir is None:
                conflicts.append(
                    "transforms enabled without artifact_dir (builds persist "
                    "artifacts and require an artifact store; set artifact_dir)"
                )

            # The MCP endpoint exposes the warm-session read/run/author surface.
            # With principal auth each tool call runs as its caller and is
            # checked against the notebook scopes; without it, it would hand
            # every reachable client full notebook control.
            if self.mcp_enabled and not self.principal_auth_enabled:
                conflicts.append(
                    "mcp_enabled=True with deployment_mode='service' and no principal "
                    "auth (the MCP endpoint grants session control and would have no "
                    "caller to check; set auth_mode='trusted_proxy' or 'api_key')"
                )

            if conflicts:
                raise ValueError(
                    "Deployment mode coherence error: deployment_mode='service' "
                    "is incompatible with:\n  - " + "\n  - ".join(conflicts)
                )
            return self

        if self.deployment_mode != "personal":
            return self

        conflicts: list[str] = []
        if self.auth_mode == "trusted_proxy":
            conflicts.append(
                "auth_mode='trusted_proxy' (personal mode has no upstream "
                "proxy; set auth_mode='none' or switch to service mode)"
            )
        if self.auth_mode == "api_key":
            conflicts.append(
                "auth_mode='api_key' (personal mode is single-user and binds to "
                "loopback; authenticating yourself to your own machine buys "
                "nothing. Switch to service mode to serve multiple accounts)"
            )
        if self.multi_tenant_enabled:
            conflicts.append(
                "multi_tenant_enabled=True (personal mode is single-user; "
                "tenants only apply in service mode)"
            )
        if self.require_tenant_header:
            conflicts.append(
                "require_tenant_header=True (personal mode has no tenants to require a header for)"
            )

        if conflicts:
            raise ValueError(
                "Deployment mode coherence error: deployment_mode='personal' "
                "is incompatible with:\n  - " + "\n  - ".join(conflicts)
            )
        return self

    def validate_personal_mode_binding(self) -> None:
        """Refuse a non-loopback bind in personal mode, which enables writes.

        Raises:
            ValueError: Unless ``allow_remote_clients_in_personal`` is set.
        """
        if self.deployment_mode != "personal":
            return

        loopback_hosts = {"127.0.0.1", "localhost", "::1"}
        is_loopback = self.host in loopback_hosts

        if not is_loopback and not self.allow_remote_clients_in_personal:
            raise ValueError(
                f"Personal mode binding to '{self.host}' is unsafe. "
                f"Personal mode enables write endpoints (artifacts, uploads). "
                f"Either bind to 127.0.0.1/localhost, or set "
                f"allow_remote_clients_in_personal=True if you have firewall protection."
            )

    def artifact_gc_policy(self) -> dict[str, Any]:
        """Return the configured retention as ``garbage_collect`` kwargs; 0 means off."""
        return {
            "max_idle_days": self.artifact_gc_max_idle_days or None,
            "max_bytes": self.artifact_gc_max_bytes or None,
            "min_idle_seconds": self.artifact_gc_min_idle_seconds,
        }

    @property
    def writes_enabled(self) -> bool:
        """Check if write endpoints are enabled (personal mode only)."""
        return self.deployment_mode == "personal"

    @property
    def principal_auth_enabled(self) -> bool:
        """Whether requests carry an authenticated principal to authorize against.

        Gates must ask this, not ``auth_mode == "trusted_proxy"``: ``api_key``
        produces the same ``Principal``, and a mode-specific gate opens under it.
        """
        return self.auth_mode in ("trusted_proxy", "api_key")

    @property
    def server_transforms_enabled(self) -> bool:
        """Check if server-mode transforms are enabled."""
        return self.deployment_mode == "service" and self.transforms_config.get("enabled", False)

    @property
    def transforms_runtime_enabled(self) -> bool:
        """Whether this server executes transform builds itself.

        Always in personal mode (otherwise materialize would park in ``building``);
        service mode needs ``transforms_config`` enabled.
        """
        return self.server_transforms_enabled or self.writes_enabled

    @property
    def max_transform_output_bytes(self) -> int:
        """Get max transform output size in bytes."""
        return self.build_runner_default_max_output

    def create_metadata_dialect(self):
        """Build the artifact store's metadata backend, or None for SQLite.

        Raises
        ------
        ValueError
            If the DSN is not postgresql or psycopg is missing; raised at startup
            so the store never accepts work it cannot keep.
        """
        if not self.artifact_metadata_dsn:
            return None

        dsn = self.artifact_metadata_dsn
        scheme = dsn.split("://", 1)[0].lower() if "://" in dsn else ""
        if scheme not in ("postgres", "postgresql"):
            raise ValueError(
                f"artifact_metadata_dsn must be a postgresql:// URL, got {scheme or dsn!r}. "
                "Leave it unset to keep the default SQLite metadata store."
            )

        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "artifact_metadata_dsn is set but the postgres extra is not "
                "installed. Install strata-notebook[postgres]."
            ) from exc

        from strata.sql_backend import PostgresDialect

        return PostgresDialect(dsn)

    def create_blob_store(self):
        """Create the artifact blob store for ``artifact_blob_backend``.

        Raises:
            ValueError: If the backend's bucket, container or ``artifact_dir`` is unset.
        """
        from strata.blob_store import (
            AzureBlobStore,
            GCSBlobStore,
            LocalBlobStore,
            S3BlobStore,
        )

        backend = self.artifact_blob_backend.lower()

        if backend == "s3":
            if not self.artifact_s3_bucket:
                raise ValueError("S3 blob backend requires artifact_s3_bucket configuration")
            return S3BlobStore.from_config(
                self,
                bucket=self.artifact_s3_bucket,
                prefix=self.artifact_s3_prefix,
            )

        if backend == "gcs":
            if not self.artifact_gcs_bucket:
                raise ValueError("GCS blob backend requires artifact_gcs_bucket configuration")
            return GCSBlobStore.from_config(
                self,
                bucket=self.artifact_gcs_bucket,
                prefix=self.artifact_gcs_prefix,
            )

        if backend == "azure":
            if not self.artifact_azure_container:
                raise ValueError(
                    "Azure blob backend requires artifact_azure_container configuration"
                )
            return AzureBlobStore.from_config(
                self,
                container_name=self.artifact_azure_container,
                prefix=self.artifact_azure_prefix,
            )

        # Default: local filesystem
        if self.artifact_dir is None:
            raise ValueError("Local blob store requires artifact_dir in configuration")
        blobs_dir = self.artifact_dir / "blobs"
        return LocalBlobStore(blobs_dir)

    def get_build_qos_config(self):
        """Create a ``BuildQoSConfig`` from the ``build_qos_*`` settings."""
        from strata.transforms.build_qos import BuildQoSConfig

        return BuildQoSConfig(
            interactive_slots=self.build_qos_interactive_slots,
            bulk_slots=self.build_qos_bulk_slots,
            per_tenant_interactive=self.build_qos_per_tenant_interactive,
            per_tenant_bulk=self.build_qos_per_tenant_bulk,
            interactive_queue_timeout=self.build_qos_interactive_timeout,
            bulk_queue_timeout=self.build_qos_bulk_timeout,
            per_tenant_timeout=self.build_qos_per_tenant_timeout,
            bytes_per_day_limit=self.build_qos_bytes_per_day,
            classify_by_estimated_bytes=self.build_qos_bulk_bytes_threshold,
            classify_by_input_count=self.build_qos_bulk_inputs_threshold,
        )

    @classmethod
    def load(cls, **overrides) -> StrataConfig:
        """Load configuration with precedence: defaults < pyproject.toml < env vars < overrides."""
        file_config = _load_from_pyproject()
        env_config = _get_env_overrides()

        # Documented precedence is pyproject < env, but file_config goes in as init kwargs, which
        # pydantic-settings ranks ABOVE env. Drop any pyproject key a STRATA_* env var overrides so
        # env wins.
        env_var_names = {name.upper() for name in os.environ}

        # A field with a validation_alias answers to more than one env name, so
        # the STRATA_{KEY} rule below cannot see all of them. gcs_project_id is
        # the legacy spelling of gcs_default_bucket_location: fold it into the
        # current name first (warning, since it names something it never set),
        # and record every env name that should shadow it.
        _ALIASED_FILE_KEYS = {
            "gcs_project_id": (
                "gcs_default_bucket_location",
                ("STRATA_GCS_PROJECT_ID", "STRATA_GCS_DEFAULT_BUCKET_LOCATION"),
            ),
        }
        extra_shadow_names: dict[str, tuple[str, ...]] = {}
        for legacy, (current, shadowing) in _ALIASED_FILE_KEYS.items():
            if legacy in file_config:
                logger.warning(
                    "[tool.strata] %s is the old name for %s and does not set a "
                    "GCP project; rename it.",
                    legacy,
                    current,
                )
                file_config.setdefault(current, file_config.pop(legacy))
            if current in file_config:
                extra_shadow_names[current] = shadowing

        for key in list(file_config):
            names = (f"STRATA_{key.upper()}", *extra_shadow_names.get(key, ()))
            if any(name in env_var_names for name in names):
                del file_config[key]

        # Deep-merge nested dict configs so an env override of one key (e.g.
        # STRATA_CATALOG_URI → catalog_properties.uri) doesn't wipe sibling keys
        # set in pyproject (type, warehouse, …).
        for nested_key in ("catalog_properties",):
            file_nested = file_config.get(nested_key)
            env_nested = env_config.get(nested_key)
            if isinstance(file_nested, dict) and isinstance(env_nested, dict):
                env_config[nested_key] = {**file_nested, **env_nested}

        # Merge: defaults < pyproject.toml < env vars < overrides
        merged = {**file_config, **env_config, **overrides}

        # Parse ACL config. Accept both [tool.strata.acl] and the documented
        # [tool.strata.acl_config]; both use the deny/allow shape and must go
        # through _parse_acl_config (the model fields are deny_rules/allow_rules,
        # so a raw acl_config dict would silently drop its rules).
        acl_raw = merged.pop("acl", None)
        if acl_raw is None and isinstance(merged.get("acl_config"), dict):
            acl_raw = merged.pop("acl_config")
        if acl_raw is not None:
            merged["acl_config"] = _parse_acl_config(acl_raw)

        # Store transforms config from [tool.strata.transforms]. Merge with any
        # env-derived transforms_config (e.g. STRATA_TRANSFORMS_ENABLED=true) so
        # the env toggle isn't lost when a pyproject block exists; env keys win.
        if "transforms" in merged:
            pyproject_transforms = merged.pop("transforms")
            env_transforms = merged.get("transforms_config")
            if isinstance(env_transforms, dict):
                merged["transforms_config"] = {**pyproject_transforms, **env_transforms}
            else:
                merged["transforms_config"] = pyproject_transforms

        return cls(**merged)

    @property
    def server_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def get_timeout_config(self) -> dict:
        """Return the timeout settings grouped by category."""
        return {
            "planning": {
                "plan_timeout_seconds": self.plan_timeout_seconds,
            },
            "scanning": {
                "scan_timeout_seconds": self.scan_timeout_seconds,
            },
            "qos_queue": {
                "interactive_queue_timeout": self.interactive_queue_timeout,
                "bulk_queue_timeout": self.bulk_queue_timeout,
            },
            "fetching": {
                "fetch_timeout_seconds": self.fetch_timeout_seconds,
            },
            "s3": {
                "s3_connect_timeout_seconds": self.s3_connect_timeout_seconds,
                "s3_request_timeout_seconds": self.s3_request_timeout_seconds,
            },
        }

    def get_s3_filesystem(self):
        """Create a PyArrow S3FileSystem from the ``s3_*`` settings."""
        import pyarrow.fs as pafs

        kwargs = {}

        if self.s3_region:
            kwargs["region"] = self.s3_region

        if self.s3_access_key and self.s3_secret_key:
            kwargs["access_key"] = self.s3_access_key
            kwargs["secret_key"] = self.s3_secret_key

        if self.s3_endpoint_url:
            kwargs["endpoint_override"] = self.s3_endpoint_url

        if self.s3_anonymous:
            kwargs["anonymous"] = True

        kwargs["connect_timeout"] = self.s3_connect_timeout_seconds
        kwargs["request_timeout"] = self.s3_request_timeout_seconds

        return pafs.S3FileSystem(**kwargs)

    def configure_arrow_memory_pool(self) -> str | None:
        """Set PyArrow's process-wide memory pool; call once at startup before Arrow work.

        Returns:
            The pool's name, or None when ``arrow_memory_pool`` is unset.

        Raises:
            ValueError: If the pool is unknown or not available in this build.
        """
        import pyarrow as pa

        if self.arrow_memory_pool is None:
            return None

        pool_name = self.arrow_memory_pool.lower()

        if pool_name == "default":
            return pa.default_memory_pool().backend_name

        if pool_name == "system":
            pa.set_memory_pool(pa.system_memory_pool())
            return "system"

        if pool_name == "jemalloc":
            try:
                pool = pa.jemalloc_memory_pool()
                pa.set_memory_pool(pool)
                return "jemalloc"
            except Exception as e:
                raise ValueError(f"jemalloc memory pool not available: {e}") from e

        if pool_name == "mimalloc":
            try:
                pool = pa.mimalloc_memory_pool()
                pa.set_memory_pool(pool)
                return "mimalloc"
            except Exception as e:
                raise ValueError(f"mimalloc memory pool not available: {e}") from e

        raise ValueError(
            f"Unknown memory pool: {self.arrow_memory_pool}. "
            f"Options: default, system, jemalloc, mimalloc"
        )


def _get_env_overrides() -> dict[str, Any]:
    """Collect env overrides pydantic-settings cannot express on its own.

    AWS_* and GOOGLE_APPLICATION_CREDENTIALS fall back behind their STRATA_*
    names; STRATA_CATALOG_URI and STRATA_TRANSFORMS_ENABLED fold into nested dicts.
    """
    overrides: dict[str, Any] = {}

    # S3 configuration (prefer STRATA_* but fall back to AWS_* for compatibility)
    if s3_region := os.environ.get("STRATA_S3_REGION") or os.environ.get("AWS_REGION"):
        overrides["s3_region"] = s3_region

    if s3_access_key := os.environ.get("STRATA_S3_ACCESS_KEY") or os.environ.get(
        "AWS_ACCESS_KEY_ID"
    ):
        overrides["s3_access_key"] = s3_access_key

    if s3_secret_key := os.environ.get("STRATA_S3_SECRET_KEY") or os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    ):
        overrides["s3_secret_key"] = s3_secret_key

    # GCS credentials (prefer STRATA_* but fall back to Google standard)
    if gcs_credentials := os.environ.get("STRATA_GCS_CREDENTIALS_JSON") or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    ):
        overrides["gcs_credentials_json"] = gcs_credentials

    # Catalog URI (for PostgreSQL or other SQL backends)
    # Example: postgresql://user:pass@localhost:5432/iceberg_catalog
    if catalog_uri := os.environ.get("STRATA_CATALOG_URI"):
        if "catalog_properties" not in overrides:
            overrides["catalog_properties"] = {}
        overrides["catalog_properties"]["uri"] = catalog_uri

    if os.environ.get("STRATA_TRANSFORMS_ENABLED", "").lower() == "true":
        if "transforms_config" not in overrides:
            overrides["transforms_config"] = {}
        overrides["transforms_config"]["enabled"] = True

    return overrides
