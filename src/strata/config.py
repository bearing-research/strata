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

# ---------------------------------------------------------------------------
# ACL Configuration Types
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)


class AclRule(BaseModel):
    """Single ACL rule for access control.

    Rules are matched in order. A rule matches if:
    - Principal matches (or rule principal is "*" for any)
    - Tenant matches (if specified in rule)
    - At least one table pattern matches

    Attributes:
        principal: Principal ID pattern ("*" for any principal)
        tenant: Optional tenant ID (None means any tenant)
        tables: Tuple of table patterns (glob-style, e.g., "file:db.*")
    """

    model_config = ConfigDict(frozen=True)

    principal: str = "*"
    tenant: str | None = None
    tables: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reject_unmatchable_rule(self) -> AclRule:
        """A rule with no table patterns can never match — reject it loudly.

        ``tables`` defaulted to ``()`` and the matcher returns False on an
        empty tuple, so ``{ principal = "bob" }`` — the natural way to write
        "deny bob everything" — was silently inert. For a deny rule that fails
        OPEN, and ``validate_mode_coherence`` still counted it as "acl
        configured", so the operator got a clean boot and a false sense of
        protection.

        Defaulting empty to "all tables" would fix deny rules but silently
        widen every *allow* rule written the same way, so the safe reading is
        that this is a configuration error: say so, and let the operator write
        ``tables = ["*"]`` when that is what they mean.
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
    """Access control list configuration.

    ACL evaluation order:
    1. Deny rules are checked first - if any match, access is denied
    2. Allow rules are checked - if any match, access is allowed
    3. Default action is applied (allow or deny)

    Attributes:
        default: Default action when no rules match ("allow" or "deny")
        deny_rules: List of deny rules (checked first)
        allow_rules: List of allow rules (checked second)

    Rules are written as ``deny`` / ``allow`` in both pyproject and
    ``STRATA_ACL_CONFIG``, so those are accepted as aliases for the field
    names. Unknown keys are rejected rather than ignored: the failure mode of
    ignoring one is an ACL that boots clean and enforces nothing.
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
    """Parse ACL configuration from pyproject.toml [tool.strata.acl] section.

    Expected format:
        [tool.strata.acl]
        default = "deny"

        deny = [
          { principal = "*", tables = ["file:finance.*", "s3:pii.*"] }
        ]

        allow = [
          { principal = "bi-dashboard", tables = ["file:db.*"] },
          { tenant = "data-platform", tables = ["file:analytics.*"] }
        ]

    Args:
        raw: Dictionary from pyproject.toml acl section

    Returns:
        Parsed AclConfig object

    This hand-unpacked ``deny``/``allow`` itself, which meant the wire shape
    was understood *here* and not by the model. Anything reaching AclConfig by
    another route — ``STRATA_ACL_CONFIG`` goes straight to pydantic's env
    source, never through this function — had its rules dropped. The model
    owns the shape now, so every path agrees.
    """
    return AclConfig.model_validate(raw)


class StrataConfig(BaseSettings):
    """Configuration for Strata server and client.

    Configuration is loaded from pyproject.toml [tool.strata] section,
    environment variables (STRATA_* prefix), and programmatic overrides.

    Precedence: defaults < pyproject.toml < env vars < overrides

    Example pyproject.toml:
        [tool.strata]
        host = "0.0.0.0"
        port = 8765
        cache_dir = "/tmp/strata-cache"
        max_cache_size_bytes = 10737418240
        cache_granularity = "row_group_projection"  # or "row_group"
        batch_size = 65536
        fetch_parallelism = 4  # Max concurrent row group fetches
        catalog_name = "default"

        # Resource limits (backpressure)
        max_concurrent_scans = 100
        max_tasks_per_scan = 1000
        plan_timeout_seconds = 30.0
        scan_timeout_seconds = 300.0
        max_response_bytes = 536870912  # 512 MB

        # Metadata persistence
        metadata_db = "/var/lib/strata/meta.sqlite"

        # S3 storage backend (optional)
        s3_region = "us-east-1"
        s3_endpoint_url = "http://localhost:9000"  # For MinIO/LocalStack
        s3_anonymous = false  # Set true for public buckets

        [tool.strata.catalog_properties]
        type = "sql"
        uri = "sqlite:///catalog.db"

    Cache granularity options:
        - row_group_projection: Cache per row-group + projection (default, finest)
        - row_group: Cache per row-group only, project on read (coarser, reuses cache)
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

    # Resource limits (backpressure)
    max_concurrent_scans: Annotated[int, Field(ge=1)] = 100
    max_tasks_per_scan: Annotated[int, Field(ge=1)] = 1000
    plan_timeout_seconds: Annotated[float, Field(gt=0)] = 30.0
    scan_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0
    max_response_bytes: Annotated[int, Field(gt=0)] = 512 * 1024 * 1024  # 512 MB
    # How long a completed/abandoned stream's state lingers before cleanup (a
    # memory/resource knob; also lets tests use a short TTL via config instead of
    # mutating server state).
    stream_state_ttl_seconds: Annotated[float, Field(gt=0)] = 300.0

    # How other nodes reach this one, e.g. "https://strata-3.internal:8765".
    # Unset means single-node: nothing is written to or read from the stream
    # ownership table, so the common case pays nothing.
    #
    # Setting it is the operator asserting two things: that this deployment
    # runs several nodes behind one address, and that this URL actually
    # reaches this node. Streams cannot move between nodes -- a live
    # asyncio.Task and an in-memory ReadPlan are not shareable -- so a node
    # asked for someone else's stream redirects to the owner instead of
    # returning a 404 that is indistinguishable from "expired".
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

    # Metadata database
    metadata_db: Path | None = None

    # S3 settings
    s3_region: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_endpoint_url: str | None = None
    s3_anonymous: bool = False

    # Memory pool settings
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

    # Fetch timeout settings
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
    # Note: per-tenant admission defaults come from interactive_slots / bulk_slots
    # (wired into the tenant registry at startup). The former default_tenant_*
    # fields were never read and were removed (issue #185).

    # Trusted proxy authentication settings
    auth_mode: Literal["none", "trusted_proxy", "api_key"] = "none"
    proxy_token_header: str = "X-Strata-Proxy-Token"
    proxy_token: str | None = None
    principal_header: str = "X-Strata-Principal"
    scopes_header: str = "X-Strata-Scopes"
    hide_forbidden_as_not_found: bool = True

    # Opt-in: let authenticated clients WRITE in service mode (put / set_name /
    # set_alias / tags), scoped to the caller's tenant and gated by the
    # `artifacts:write` scope. Default off — service mode is read-only unless this
    # is set. Requires trusted-proxy auth (writes must be attributable). For the
    # shared-research-store deployment (team = tenant, principal = author).
    service_writes_enabled: bool = False

    # Access control list configuration
    acl_config: AclConfig = Field(default_factory=AclConfig)

    # Deployment mode settings.
    #
    # Default is ``personal`` — the common case. A first-time
    # ``uv run strata-notebook`` boots into a single-user, loopback-only
    # configuration that just works. Multi-user and multi-tenant
    # deployments must opt in explicitly via ``deployment_mode="service"``
    # plus the matching auth / artifact settings; that flow has its own
    # coherence checks (see ``validate_mode_coherence``) so production
    # operators get clear errors if anything's misconfigured.
    deployment_mode: Literal["service", "personal"] = "personal"
    allow_remote_clients_in_personal: bool = False
    # Extra browser origins allowed to make cross-origin calls to this server.
    #
    # Same-origin is always allowed, so the bundled frontend needs nothing here.
    # This exists for `npm run dev`, where Vite serves the UI from another port
    # and talks to this server via VITE_STRATA_URL — set it to that dev origin,
    # e.g. ["http://localhost:5173"].
    #
    # It defaults to EMPTY on purpose. The server used to send
    # Access-Control-Allow-Origin: * , which let any page the user happened to
    # visit drive the loopback API: personal mode has no auth, so a page could
    # enumerate notebooks, add a cell containing arbitrary Python and execute
    # it. Binding to loopback is no defence, because the browser runs there too.
    cors_allow_origins: list[str] = []
    # Mount the MCP server at ``/mcp`` so an external coding agent (Claude Code,
    # etc.) can drive the live notebook session over streamable HTTP. Opt-in and
    # PERSONAL MODE ONLY — it exposes the same warm-session read/run/author
    # surface the loopback REST API does, with no per-request auth, so service
    # deployments must not turn it on (enforced in ``validate_mode_coherence``).
    # Requires the ``[mcp]`` extra; without it the flag warns and no-ops.
    mcp_enabled: bool = False
    # Optional request header that identifies the calling user when a personal
    # mode deployment is fronted by an authenticating proxy (Cloudflare Access,
    # Pomerium, etc.). When set, notebooks are stamped with the caller's
    # identity on create and the discover/delete endpoints scope to it. When
    # unset, the deployment behaves like a single-user instance — the default
    # for a developer running on localhost.
    personal_mode_user_header: str | None = None
    # Origins allowed to embed a notebook's app view in an ``<iframe>`` (e.g.
    # a dashboard or wiki on another host). Sets ``Content-Security-Policy:
    # frame-ancestors 'self' <origins>`` on every response. Empty (the default)
    # means same-origin only — external embedding is opt-in, and unlike the old
    # no-header behavior a stray page can no longer frame Strata. Accepts a JSON
    # array or a comma-separated list; each entry is an origin
    # (``https://analytics.example.com``) or ``*`` to allow any host.
    embed_frame_ancestors: Annotated[list[str], NoDecode] = Field(default_factory=list)
    artifact_dir: Path | None = None
    # Builds stuck in 'building' longer than this are demoted to failed at
    # startup (zombie sweep) — they can never serve data and would otherwise
    # linger in the store forever.
    artifact_zombie_build_timeout_seconds: Annotated[float, Field(gt=0)] = 3600.0
    # Registry aliases that require approval: moves/deletes of these aliases
    # (e.g. "champion") land in a pending queue instead of applying, and an
    # explicit approve applies them. Empty (the default) = no gating.
    registry_protected_aliases: Annotated[list[str], NoDecode] = Field(default_factory=list)
    notebook_storage_dir: Path = Field(
        default_factory=lambda: Path.home() / ".strata" / "notebooks"
    )
    notebook_python_versions: list[str] = Field(default_factory=discover_installed_python_minors)

    # Point the ambient `strata` client injected into notebook cells at a REMOTE
    # shared store instead of this local notebook server. Lets a team of
    # researchers publish/consume datasets against one central deployment (the
    # shared research store). `notebook_remote_store_headers` carries the auth the
    # remote store needs (e.g. the trusted-proxy identity/token, or a bearer
    # token) — set via env so secrets stay out of committed config. Unset → the
    # ambient client targets the local server as before.
    notebook_remote_store_url: str | None = None
    notebook_remote_store_headers: dict[str, str] = Field(default_factory=dict)
    # Consult the remote store on a LOCAL cache miss, so a colleague's
    # expensive cell becomes your instant result. Distinct from the knob above,
    # which only redirects a cell's ambient client: that is explicit publish
    # (someone names a dataset and pushes it), and nobody names their
    # intermediate results — which is exactly where the recomputation is.
    #
    # Opt-in, and separate from the URL, because it is a real behaviour change
    # and not only a performance one: it puts bytes another machine produced
    # into your local store, and adds a network round-trip to the miss path of
    # every cell. Wanting a shared store to publish to is not the same as
    # wanting one to silently source results from.
    notebook_team_cache_enabled: bool = False

    # AI/LLM assistant settings (OpenAI-compatible API)
    ai_base_url: str | None = None
    ai_model: str | None = None
    ai_api_key: str | None = None
    ai_max_context_tokens: Annotated[int, Field(gt=0)] = 100_000
    ai_max_output_tokens: Annotated[int, Field(gt=0)] = 4096
    ai_timeout_seconds: Annotated[float, Field(gt=0)] = 60.0
    # How long an agent destructive-tool confirmation waits for the user
    # before being treated as a decline.
    ai_approval_timeout_seconds: Annotated[float, Field(gt=0)] = 120.0

    # Artifact blob storage backend configuration
    # Metadata backend for the artifact store. Unset means SQLite in
    # artifact_dir, which is what personal mode wants and what every existing
    # deployment already has. A DSN moves the store's system of record to a
    # shared server, which is what lets more than one node share one store.
    # Blobs are configured separately via artifact_blob_backend; a shared
    # database paired with local blobs is only coherent on a single machine,
    # so the two are validated together below.
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
    # Named for what it does. PyArrow's GcsFileSystem has no project parameter
    # at all, so the old STRATA_GCS_PROJECT_ID never set one — it fed whatever
    # it was given to ``default_bucket_location``, a GCS location like ``US``
    # or ``europe-west1``. The old name stays accepted so a deployment setting
    # it keeps the behaviour it had; ``validate_gcs_settings`` says what it
    # actually controls.
    # Both entries carry the STRATA_ prefix deliberately. validation_alias
    # *replaces* env_prefix rather than combining with it, so a bare
    # "gcs_project_id" here would make the unprefixed GCS_PROJECT_ID live
    # config — and in a GCP deployment that variable is ambient, which would
    # feed a project id into the location field exactly as before. No other
    # setting in this class is reachable without the prefix.
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
    # HMAC secret for signing pull-model build URLs (env STRATA_TRANSFORM_SIGNING_SECRET).
    # If unset, a random per-process secret is used — fine for single-instance dev,
    # but signed URLs then become invalid on restart and differ across replicas.
    # Set a stable value for any multi-replica or restart-surviving deployment.
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
        # Ensure cache_dir exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set default metadata_db if not specified
        if self.metadata_db is None:
            self.metadata_db = Path.home() / ".strata" / "meta.sqlite"
        # Ensure metadata_db parent directory exists
        if self.metadata_db is not None:
            self.metadata_db.parent.mkdir(parents=True, exist_ok=True)

        # Set default artifact_dir for personal mode
        if self.artifact_dir is None and self.deployment_mode == "personal":
            self.artifact_dir = Path.home() / ".strata" / "artifacts"

        # Ensure artifact_dir exists in personal mode
        if self.deployment_mode == "personal" and self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Ensure the default notebook storage directory exists.
        self.notebook_storage_dir.mkdir(parents=True, exist_ok=True)

        return self

    @model_validator(mode="after")
    def validate_adaptive_ranges(self) -> StrataConfig:
        """Validate the adaptive controller's bounds and its starting point.

        The starting-point checks matter as much as the min<=max ones. The
        controller seeds each tier from the configured slot count and then
        clamps every adjustment into ``[min, max]``; starting outside that
        range means the very first adjustment jumps to a bound, and a decrease
        requested under load can land *above* where it started. Rejecting the
        configuration is the honest fix — the operator picked those slot
        counts on purpose.
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
        """The team cache needs a store to be a cache *of*.

        Not a deployment-mode check: the notebook that reads a team store is
        normally a personal-mode process on someone's laptop, pointed at a
        service-mode store elsewhere. It is the pairing that has to hold, in
        any mode.

        Enabled without a URL is silently inert — every lookup would have
        nowhere to go, so every cell would recompute and the operator would
        conclude the shared cache does not work rather than that it was never
        switched on. That is a worse outcome than refusing to start.
        """
        if self.notebook_team_cache_enabled and not self.notebook_remote_store_url:
            raise ValueError(
                "notebook_team_cache_enabled=True without notebook_remote_store_url "
                "(there is no store to look results up in, so every lookup would "
                "miss and every cell would recompute; set notebook_remote_store_url)"
            )
        return self

    @model_validator(mode="after")
    def warn_on_gcs_project_id(self) -> StrataConfig:
        """Say what STRATA_GCS_PROJECT_ID actually controls.

        It never set a project: PyArrow's GcsFileSystem takes no project
        parameter, so the value went to ``default_bucket_location`` — a GCS
        location like ``US``. An operator who read the name and set a project
        id has a bogus location configured, which is inert until something
        creates a bucket, at which point it is not. The setting keeps working
        under either name; this is the only place that can point out the two
        do not mean the same thing.
        """
        import os

        # Only when the old name is what supplied the value: with both set the
        # new one wins and the old is already being ignored, so saying "rename
        # it" would be advice about a setting that is doing nothing.
        if os.environ.get("STRATA_GCS_PROJECT_ID") and not os.environ.get(
            "STRATA_GCS_DEFAULT_BUCKET_LOCATION"
        ):
            logger.warning(
                "STRATA_GCS_PROJECT_ID does not set a GCP project — GcsFileSystem "
                "has no project parameter. Its value is used as the default bucket "
                "location (a GCS location such as 'US' or 'europe-west1'). Rename "
                "it to STRATA_GCS_DEFAULT_BUCKET_LOCATION, and check the value is "
                "a location rather than a project id."
            )
        return self

    @model_validator(mode="after")
    def validate_mode_coherence(self) -> StrataConfig:
        """Reject deployment-mode combinations that indicate misconfiguration.

        Personal mode is a single-user local deployment: one identity, no tenant
        dimension, no upstream proxy. Turning on trusted-proxy auth or
        multi-tenancy in personal mode doesn't do anything useful and almost
        always means the operator pulled flags from a service-mode config by
        mistake. Failing fast at startup beats a confusing runtime.
        """
        # Service mode: reject configs whose security/build intent is silently
        # inert.
        if self.deployment_mode == "service":
            conflicts: list[str] = []

            # Multi-tenancy is an access-control boundary. Without auth the tenant
            # header is unauthenticated and spoofable, and direct artifact reads
            # aren't tenant-filtered — so multi-tenancy requires trusted-proxy auth
            # to mean anything.
            if self.multi_tenant_enabled and not self.principal_auth_enabled:
                conflicts.append(
                    f"multi_tenant_enabled=True with auth_mode={self.auth_mode!r} "
                    "(the tenant header is unauthenticated and spoofable, and "
                    "reads aren't tenant-filtered without auth; set "
                    "auth_mode='trusted_proxy' or 'api_key')"
                )

            # Trusted-proxy auth without a shared token is no auth at all:
            # verify_proxy_token() returns True when no token is configured, so
            # any client that can reach Strata can spoof the principal/scope
            # headers. Require the token (it's free even for network-isolated
            # deployments) rather than silently trusting every caller.
            if self.auth_mode == "trusted_proxy" and not self.proxy_token:
                conflicts.append(
                    "auth_mode='trusted_proxy' without proxy_token (the token is "
                    "unset, so every request is accepted and principal/scope "
                    "headers can be spoofed; set proxy_token)"
                )

            # Authenticated write-back must be attributable: writes are stamped
            # with the caller's principal/tenant, which only exist under auth.
            # Still trusted-proxy only, deliberately: the write path stamps the
            # caller's identity into stored artifacts, and that is a wider claim
            # than a read gate. Named the mode it actually saw, because reporting
            # 'none' to an api_key deployment sends the operator looking for a
            # setting that is already correct.
            if self.service_writes_enabled and self.auth_mode != "trusted_proxy":
                conflicts.append(
                    f"service_writes_enabled=True with auth_mode={self.auth_mode!r} "
                    "(writes are attributed to and scoped by the caller's "
                    "principal/tenant, which require trusted-proxy auth; set "
                    "auth_mode='trusted_proxy')"
                )

            # personal_mode_user_header is a personal-mode proxy shim; service
            # mode uses `X-Strata-Principal` via the trusted-proxy pipeline.
            if self.personal_mode_user_header:
                conflicts.append(
                    "personal_mode_user_header (a personal-mode proxy shim; service "
                    "mode uses auth_mode='trusted_proxy' with X-Strata-Principal)"
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

            # A shared metadata store paired with local blobs is only coherent
            # on one machine: node B resolves an artifact's metadata from the
            # shared database, then looks for bytes that only exist on node A's
            # disk. The read fails at fetch time, long after the request that
            # created it looked successful. Moving the metadata off SQLite is
            # done specifically to run more than one node, so this combination
            # is always a misconfiguration in service mode.
            if self.artifact_metadata_dsn and self.artifact_blob_backend == "local":
                conflicts.append(
                    "artifact_metadata_dsn with artifact_blob_backend='local' "
                    "(the metadata is shared across nodes but the blobs are not, "
                    "so another node resolves an artifact and then cannot read "
                    "its bytes; set artifact_blob_backend to s3, gcs, or azure)"
                )

            # ACL rules are only evaluated when a principal was authenticated.
            # Configured rules without auth would be silently ignored — an
            # operator who wrote a deny rule would believe they were protected.
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

            # Transform builds persist artifacts, which require an artifact store
            # (its metadata DB lives under artifact_dir). Without it, every build
            # would fail at runtime — reject at startup instead.
            if self.server_transforms_enabled and self.artifact_dir is None:
                conflicts.append(
                    "transforms enabled without artifact_dir (builds persist "
                    "artifacts and require an artifact store; set artifact_dir)"
                )

            # The MCP endpoint exposes the warm-session read/run/author surface
            # with no per-request auth — safe only behind a loopback personal
            # deployment. In service mode it would hand every reachable client
            # full notebook control, so refuse the combination outright.
            if self.mcp_enabled:
                conflicts.append(
                    "mcp_enabled=True with deployment_mode='service' (the MCP "
                    "endpoint has no per-request auth and grants full session "
                    "control; it is personal-mode only)"
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
        if self.mcp_enabled and self.personal_mode_user_header:
            # personal_mode_user_header is the proxy-fronted multi-user shim:
            # `discover` and `delete` filter by owner, and every REST notebook
            # route runs `_require_owner`. The MCP mount has no per-request
            # identity to check against, and no owner filtering anywhere — its
            # `list_notebooks` returns every open session with its path, and
            # any tool call accepts any session id. Combining the two lets one
            # user enumerate and execute code in another user's notebook,
            # which the REST API on the same server would 404.
            conflicts.append(
                "mcp_enabled=True with personal_mode_user_header set (the MCP "
                "endpoint has no per-request identity and does not filter by "
                "owner, so it would expose every user's sessions; use one or "
                "the other)"
            )

        if conflicts:
            raise ValueError(
                "Deployment mode coherence error: deployment_mode='personal' "
                "is incompatible with:\n  - " + "\n  - ".join(conflicts)
            )
        return self

    def validate_personal_mode_binding(self) -> None:
        """Validate that personal mode binding is safe.

        In personal mode, binding to non-loopback addresses exposes the server
        to the network, which is dangerous since personal mode enables writes.

        Raises:
            ValueError: If personal mode binds to non-loopback without explicit allow
        """
        if self.deployment_mode != "personal":
            return

        # Check if host is loopback
        loopback_hosts = {"127.0.0.1", "localhost", "::1"}
        is_loopback = self.host in loopback_hosts

        if not is_loopback and not self.allow_remote_clients_in_personal:
            raise ValueError(
                f"Personal mode binding to '{self.host}' is unsafe. "
                f"Personal mode enables write endpoints (artifacts, uploads). "
                f"Either bind to 127.0.0.1/localhost, or set "
                f"allow_remote_clients_in_personal=True if you have firewall protection."
            )

    @property
    def writes_enabled(self) -> bool:
        """Check if write endpoints are enabled (personal mode only)."""
        return self.deployment_mode == "personal"

    @property
    def principal_auth_enabled(self) -> bool:
        """Whether requests carry an authenticated principal to authorize against.

        Authorization gates must ask this, not ``auth_mode == "trusted_proxy"``.
        A proxy header and a bearer key produce the same ``Principal`` — they
        differ only in how it was established — so a gate written against one
        mode silently opens under the other. That is what happened when
        ``api_key`` was added: it authenticated callers and then authorized
        none of them.
        """
        return self.auth_mode in ("trusted_proxy", "api_key")

    @property
    def server_transforms_enabled(self) -> bool:
        """Check if server-mode transforms are enabled."""
        return self.deployment_mode == "service" and self.transforms_config.get("enabled", False)

    @property
    def transforms_runtime_enabled(self) -> bool:
        """Whether this server executes transform builds itself.

        Service mode requires the explicit transforms allowlist config.
        Personal mode always runs the embedded transforms (duckdb_sql)
        in-process — a single-user server that can't execute the flagship
        artifact workflow would silently park materialize requests in
        ``building`` forever.
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
            If the DSN names a scheme this does not support, or the driver for
            it is not installed. Both are raised at startup rather than at the
            first query, because a store that fails on its first write has
            already accepted work it cannot keep.
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
        """Create blob store based on configuration.

        Returns:
            BlobStore instance for artifact storage.

        Raises:
            ValueError: If required configuration is missing.
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
        """Create BuildQoSConfig from Strata configuration.

        Returns:
            BuildQoSConfig instance for initializing BuildQoS.
        """
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
        """Load configuration with precedence: defaults < pyproject.toml < env vars < overrides.

        Args:
            **overrides: Values that override all other settings

        Returns:
            StrataConfig instance
        """
        file_config = _load_from_pyproject()
        env_config = _get_env_overrides()

        # Documented precedence is pyproject < env, but file_config is passed to
        # the model as init kwargs, which pydantic-settings ranks ABOVE its env
        # source — so a key written in pyproject would otherwise shadow STRATA_*
        # (e.g. STRATA_AUTH_MODE silently ignored). Drop any pyproject key that a
        # STRATA_* env var overrides, letting the env source win.
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
        """Get all timeout-related configuration as a dictionary.

        Returns:
            Dictionary with all timeout settings organized by category.
        """
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
        """Create a PyArrow S3FileSystem from configuration.

        Returns:
            Configured S3FileSystem for reading Parquet files from S3

        Raises:
            ImportError: If pyarrow.fs is not available
        """
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

        # Apply timeout settings
        kwargs["connect_timeout"] = self.s3_connect_timeout_seconds
        kwargs["request_timeout"] = self.s3_request_timeout_seconds

        return pafs.S3FileSystem(**kwargs)

    def configure_arrow_memory_pool(self) -> str | None:
        """Configure PyArrow's global memory pool based on settings.

        This affects all PyArrow allocations in the process. Should be called
        once at server startup before any Arrow operations.

        Returns:
            The name of the configured pool, or None if no change was made.

        Raises:
            ValueError: If the specified pool is not available.
        """
        import pyarrow as pa

        if self.arrow_memory_pool is None:
            return None

        pool_name = self.arrow_memory_pool.lower()

        if pool_name == "default":
            # Use PyArrow's default (no change needed)
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


def _get_env_overrides() -> dict:
    """Get configuration overrides from environment variables.

    This function handles AWS_* fallbacks and complex parsing that
    pydantic-settings doesn't handle automatically.

    Supported environment variables with special handling:
    - AWS_REGION / STRATA_S3_REGION: S3 region (AWS fallback)
    - AWS_ACCESS_KEY_ID / STRATA_S3_ACCESS_KEY: S3 access key (AWS fallback)
    - AWS_SECRET_ACCESS_KEY / STRATA_S3_SECRET_KEY: S3 secret key (AWS fallback)
    - GOOGLE_APPLICATION_CREDENTIALS: GCS credentials fallback
    - STRATA_CATALOG_URI: Catalog database URI (merged into catalog_properties)
    """
    overrides = {}

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
        # Merge into catalog_properties
        if "catalog_properties" not in overrides:
            overrides["catalog_properties"] = {}
        overrides["catalog_properties"]["uri"] = catalog_uri

    # Server-mode transforms (complex nested config)
    if os.environ.get("STRATA_TRANSFORMS_ENABLED", "").lower() == "true":
        if "transforms_config" not in overrides:
            overrides["transforms_config"] = {}
        overrides["transforms_config"]["enabled"] = True

    return overrides
