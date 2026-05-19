/** Core notebook types — maps to Strata's artifact/transform model */

export type CellId = string

export type CellLanguage = 'python' | 'prompt' | 'markdown' | 'sql'
export type MountMode = 'ro' | 'rw'
export type WorkerBackend = 'local' | 'executor'
export type WorkerHealth = 'healthy' | 'unknown' | 'unavailable' | 'warming'
export type WorkerTransport = 'local' | 'embedded' | 'direct' | 'signed' | 'executor'

export interface MountSpec {
  name: string
  uri: string
  mode: MountMode
  pin?: string | null
}

/** SQL connection driver. Open-ended (server adds new drivers); UI
 * forms know how to render the two phase-1 drivers and fall back
 * to a generic key/value editor for anything else. */
export type ConnectionDriver = string

/** Auth values as stored: ``${VAR}`` indirection or empty (literal
 * values are scrubbed to "" before they reach disk). */
export type ConnectionAuth = Record<string, string>

export interface ConnectionSpec {
  name: string
  driver: ConnectionDriver
  /** SQLite: filesystem path. Resolved against the notebook dir
   * when relative. */
  path?: string | null
  /** Postgres / Flight SQL: connection URI. */
  uri?: string | null
  /** Postgres: search_path / role / etc. Driver-defined keys. */
  options?: Record<string, unknown> | null
  /** Auth credentials (``user``, ``password``, etc) — values are
   * ``${VAR}`` indirections; literals are blanked at write time. */
  auth?: ConnectionAuth | null
  /** Postgres: SET ROLE applied at connection open. */
  role?: string | null
  /** Postgres: search_path applied at connection open. */
  search_path?: string | null
  /** Driver-specific extras the server preserves. */
  [key: string]: unknown
}

export interface WorkerSpec {
  name: string
  backend: WorkerBackend
  runtimeId?: string | null
  config: Record<string, unknown>
}

export interface EditableWorkerSpec extends WorkerSpec {
  enabled?: boolean
}

export interface ManagedWorkerSpec extends WorkerSpec {
  enabled: boolean
}

export interface WorkerHealthHistoryEntry {
  checkedAt: number
  health: WorkerHealth
  error?: string | null
  durationMs?: number | null
}

export interface WorkerCatalogEntry extends WorkerSpec {
  health: WorkerHealth
  source?: 'builtin' | 'notebook' | 'server' | 'referenced'
  allowed?: boolean
  enabled?: boolean
  transport?: WorkerTransport
  healthUrl?: string | null
  healthCheckedAt?: number | null
  lastError?: string | null
  healthHistory?: WorkerHealthHistoryEntry[]
  probeCount?: number
  healthyProbeCount?: number
  unavailableProbeCount?: number
  unknownProbeCount?: number
  consecutiveFailures?: number
  lastHealthyAt?: number | null
  lastUnavailableAt?: number | null
  lastUnknownAt?: number | null
  lastStatusChangeAt?: number | null
  lastProbeDurationMs?: number | null
}

export interface CellAnnotations {
  /** Human-readable cell name from @name annotation, displayed in DAG */
  name?: string | null
  worker?: string | null
  timeout?: number | null
  env: Record<string, string>
  mounts: MountSpec[]
  /** Loop cell annotations — present only when the cell declares ``# @loop``. */
  loop?: LoopAnnotationInfo | null
  /** Variant grouping — present only when the cell declares ``# @variant``. */
  variant?: VariantAnnotationInfo | null
}

export interface VariantAnnotationInfo {
  group: string
  name: string
}

/** A single variant in a group — surfaced for tab rendering. */
export interface VariantMember {
  cellId: CellId
  name: string
  isActive: boolean
}

/** Resolved variant group state from the backend. */
export interface VariantGroup {
  group: string
  activeName: string
  activeCellId: CellId
  members: VariantMember[]
}

export interface LoopAnnotationInfo {
  maxIter: number
  carry: string
  untilExpr?: string | null
  startFromCell?: string | null
  startFromIter?: number | null
}

export type CellStatus =
  | 'idle' // never executed
  | 'queued' // waiting for upstream deps in cascade
  | 'running' // executing
  | 'ready' // has cached artifact (provenance matches)
  | 'stale' // upstream changed, provenance mismatch (coarse — see StalenessReason)
  | 'error' // execution failed

/** Fine-grained reason why a cell is stale. Multiple reasons can apply;
 *  the server returns all of them, the UI shows the most actionable one. */
export type StalenessReason =
  | 'self' // cell source was edited since last run
  | 'upstream' // cell source unchanged, but an upstream input is stale or re-ran
  | 'env' // environment (uv.lock runtime deps) changed since last run
  | 'forced' // ran with stale inputs ("Run this only") — result exists but suspect

/** State of a single input variable for a cell */
export type InputState = 'ready' | 'stale' | 'missing' | 'error'

export interface CellInput {
  /** Variable name */
  variable: string
  /** Which cell defines this variable */
  sourceCellId: CellId
  /** Current state of the artifact for this input */
  state: InputState
}

/** Content type for artifact serialization (three-tier system) */
export type ArtifactContentType =
  | 'arrow/ipc' // Tier 1: DataFrames, Tables, arrays (zero-copy fast path)
  | 'json/object' // Tier 2: Dicts, lists, JSON-safe scalars (safe, portable)
  | 'msgpack/object' // Tier 2: Dicts with bytes/datetime (safe, portable)
  | 'pickle/object' // Tier 3: Models, custom objects (unsafe — see security model)
  | 'image/png' // Display-only (plots, charts)
  | 'text/markdown' // Display-only rich text

export interface CellOutput {
  /** Content type determines how to render */
  contentType: ArtifactContentType
  /** Arrow IPC bytes decoded to row/column data for display (when contentType = arrow/ipc) */
  columns?: string[]
  rows?: Record<string, unknown>[]
  rowCount?: number
  /** Scalar/dict output (when contentType = json/scalar) */
  scalar?: unknown
  /** Inline data URL for image-like display outputs */
  inlineDataUrl?: string | null
  /** Raw markdown source for markdown display outputs */
  markdownText?: string | null
  /** Optional image dimensions for image-like display outputs */
  width?: number | null
  height?: number | null
  /** Strata artifact reference */
  artifactUri?: string
  /** Whether this came from cache */
  cacheHit?: boolean
  /** Cache load time in ms (for displaying "⚡ cached · 5ms") */
  cacheLoadMs?: number
  /** Error message if failed */
  error?: string
}

export interface Cell {
  id: CellId
  /** Source code */
  source: string
  language: CellLanguage
  /** Display order in the notebook */
  order: number
  /** Execution state */
  status: CellStatus
  /** Why the cell is stale (only present when status === 'stale') */
  stalenessReasons?: StalenessReason[]
  /** Legacy last visible output compatibility shim */
  output?: CellOutput
  /** Ordered visible outputs rendered for this cell */
  displayOutputs?: CellOutput[]
  /** Structured input status — each input with its artifact state */
  inputs: CellInput[]
  /** Cells this cell depends on (reads variables from) */
  upstreamIds: CellId[]
  /** Cells that depend on this cell */
  downstreamIds: CellId[]
  /** Variable names this cell defines */
  defines: string[]
  /** Variable names this cell references (imports from DAG) */
  references: string[]
  /** Whether this is a leaf node (no downstream consumers of its outputs) */
  isLeaf: boolean
  /** Strata provenance hash — if same as stored, result is cached */
  provenanceHash?: string
  /** Last execution timestamp */
  lastRunAt?: number
  /** Execution duration in ms */
  durationMs?: number
  /** Whether this cell is frozen (skip invalidation, pinned artifact) */
  frozen?: boolean
  /** Assertion results from this cell's execution */
  assertions?: AssertionResult[]
  /** Artifact size in bytes */
  artifactSizeBytes?: number
  /** Which executor ran this cell (only present if remote) */
  executorName?: string
  /** Name of the remote worker used for the last remote execution */
  remoteWorkerName?: string
  /** Transport used for the last remote execution */
  remoteTransport?: WorkerTransport | null
  /** Build id for the last signed remote execution */
  remoteBuildId?: string | null
  /** Last observed build state for signed remote execution */
  remoteBuildState?: string | null
  /** Structured remote execution error code, when available */
  remoteErrorCode?: string | null
  /** Effective persisted worker after notebook default + cell override */
  worker: string | null
  /** Persisted cell-level worker override from notebook.toml */
  workerOverride: string | null
  /** Effective persisted timeout after notebook default + cell override */
  timeout: number | null
  /** Persisted cell-level timeout override from notebook.toml */
  timeoutOverride: number | null
  /** Effective persisted env after notebook default + cell override */
  env: Record<string, string>
  /** Persisted cell-level env overrides from notebook.toml */
  envOverrides: Record<string, string>
  /** Effective mounts after notebook defaults + cell overrides */
  mounts: MountSpec[]
  /** Persisted cell-level overrides from notebook.toml */
  mountOverrides: MountSpec[]
  /** Source-level annotations parsed by the backend */
  annotations?: CellAnnotations
  /** Causality chain explaining why this cell is stale */
  causality?: CausalityChain
  /** Suggested package to install (when execution fails with ModuleNotFoundError) */
  suggestInstall?: string
  /** Shadow warnings from the DAG builder */
  shadowWarnings?: string[]
  /** Annotation validation diagnostics (set on open/reload, never during typing) */
  annotationDiagnostics?: AnnotationDiagnostic[]
  /** Live loop-cell progress — hydrated from WS ``cell_iteration_progress`` messages */
  loopProgress?: LoopProgress
  /** Captured stdout from the last execution (persisted so it survives reopens) */
  consoleStdout?: string
  /** Captured stderr from the last execution (persisted so it survives reopens) */
  consoleStderr?: string
  /** True when the cell's source classifies as a module cell (pure defs/classes
   * + optional literal constants). Drives the "module" pill in the UI. */
  isModuleCell?: boolean
  /** Symbols exported by this cell when it's a module cell — shown in the
   * module pill's tooltip so users see what crosses the cell boundary. */
  moduleExports?: Array<{ name: string; kind: string }>
  /** Variant group ID parsed from ``# @variant <group> <name>``. Null for
   * cells that aren't members of a group. */
  variantGroup?: string | null
  /** Variant name within ``variantGroup``. */
  variantName?: string | null
  /** False for inactive variants (shadowed in the DAG). True otherwise. */
  variantActive?: boolean
}

/** A warning or info about a source annotation. */
export interface AnnotationDiagnostic {
  severity: 'error' | 'warn' | 'info'
  code: string
  message: string
  line?: number | null
}

/** Live progress state for a loop cell, hydrated as iterations complete. */
export interface LoopProgress {
  /** 0-based index of the most recently completed iteration */
  iteration: number
  /** Safety bound from ``# @loop max_iter=N`` */
  maxIter: number
  /** Artifact URI of the most recently stored iteration */
  artifactUri?: string
  /** Content type of the stored carry artifact */
  contentType?: string
  /** True when ``@loop_until`` fired on this iteration — the loop is done */
  untilReached: boolean
  /** Duration of the most recent iteration in ms */
  iterDurationMs?: number
}

/** Causality chain — explains why a cell is stale */
export interface CausalityChain {
  reason: StalenessReason
  details: CausalityDetail[]
}

export interface CausalityDetail {
  type: 'source_changed' | 'input_changed' | 'env_changed'
  /** For source/input changes: which cell changed */
  cellId?: CellId
  /** Human-readable name of the changed cell */
  cellName?: string
  /** For input_changed: old and new artifact versions */
  fromVersion?: string
  toVersion?: string
  /** For env_changed: which package changed */
  package?: string
  fromPackageVersion?: string
  toPackageVersion?: string
}

/** Assertion result from a cell's assert statements */
export interface AssertionResult {
  /** The assertion message (from assert ..., "message") */
  message: string
  passed: boolean
  /** Expression that was asserted (source text) */
  expression?: string
  /** Actual value on failure */
  actualValue?: string
}

/** Run impact preview — what will happen if a cell is executed */
export interface ImpactPreview {
  targetCellId: CellId
  /** Upstream cells that need to run first */
  upstream: CascadeStep[]
  /** Downstream cells that will become stale */
  downstream: DownstreamImpact[]
  estimatedMs: number
}

export interface DownstreamImpact {
  cellId: CellId
  cellName: string
  currentStatus: CellStatus
  /** Status after target cell runs */
  newStatus: 'stale:upstream'
}

/** Published output — a cell's artifact exposed as a stable endpoint */
export interface PublishedOutput {
  name: string
  cellId: CellId
  mode: 'static' | 'api'
  /** Schema derived from artifact metadata */
  schema?: { columns: string[] }
  /** Last updated timestamp */
  lastUpdatedAt?: number
  /** Artifact URI */
  artifactUri?: string
}

/** Artifact lineage node — one level in the provenance chain */
export interface LineageNode {
  artifactUri: string
  artifactVersion: number
  /** Transform that produced this artifact */
  transform?: {
    executor: string
    sourceHash?: string
    cellId?: string
  }
  /** Input artifacts (recurse for full chain) */
  inputs: LineageNode[]
  /** Environment hash at time of production */
  envHash?: string
}

/** Package dependency info */
export interface DependencyInfo {
  name: string
  version: string
  specifier: string
}

export interface NotebookEnvironment {
  pythonVersion: string
  requestedPythonVersion: string
  runtimePythonVersion: string
  lockfileHash: string
  packageCount: number
  declaredPackageCount: number
  resolvedPackageCount: number
  syncState: 'unknown' | 'pending' | 'ready' | 'fallback' | 'failed'
  syncError: string | null
  syncNotice: string | null
  lastSyncedAt: number | null
  lastSyncDurationMs: number | null
  hasLockfile: boolean
  venvPython: string | null
  interpreterSource: 'unknown' | 'venv' | 'path'
}

export interface NotebookRuntimeConfig {
  deploymentMode: 'personal' | 'service'
  defaultParentPath: string
  availablePythonVersions: string[]
  defaultPythonVersion: string
  pythonSelectionFixed: boolean
}

export interface EnvironmentActionSummary {
  action: 'add' | 'remove' | 'sync' | 'import'
  packageName: string | null
  lockfileChanged: boolean
  staleCellCount: number
  timestamp: number
}

export interface EnvironmentOperation {
  id: string
  action: 'add' | 'remove' | 'sync' | 'import'
  status: 'running' | 'completed' | 'failed'
  packageName: string | null
  phase: string | null
  command: string
  durationMs: number | null
  stdout: string
  stderr: string
  stdoutTruncated: boolean
  stderrTruncated: boolean
  startedAt: number
  finishedAt: number | null
  lockfileChanged: boolean
  staleCellCount: number
  staleCellIds: string[]
  error: string | null
}

export interface EnvironmentImportPreview {
  kind: 'requirements' | 'environment_yaml'
  previewDependencies: DependencyInfo[]
  normalizedRequirements: string[]
  importedCount: number
  warnings: string[]
  additions: DependencyInfo[]
  removals: DependencyInfo[]
  unchanged: DependencyInfo[]
}

export interface Notebook {
  id: string
  name: string
  worker: string | null
  timeout: number | null
  env: Record<string, string>
  /** Per-key provenance for env. 'manual' = set via Runtime panel,
   * provider name (e.g. 'infisical') = fetched from a secret manager. */
  envSources: Record<string, string>
  /** Last secret-manager fetch error, if any. null when fetch succeeded
   * or no manager is configured. */
  envFetchError: string | null
  /** ISO-8601 timestamp of the last secret-manager fetch. */
  envFetchedAt: string | null
  /** Non-sensitive routing for the configured secret manager. Empty
   * object means no manager is configured. */
  secretManagerConfig: Record<string, string>
  workers: WorkerSpec[]
  mounts: MountSpec[]
  /** SQL connection definitions, keyed by name. Empty when the
   * notebook has no [connections.<name>] blocks. */
  connections: ConnectionSpec[]
  cells: Cell[]
  /** Resolved variant groups; one entry per group declared by ``# @variant``. */
  variantGroups: VariantGroup[]
  /** Environment info */
  environment: NotebookEnvironment
  /** Published outputs exposed as stable endpoints */
  publishedOutputs?: PublishedOutput[]
  /** Global metadata */
  createdAt: number
  updatedAt: number
}

/** Maps to Strata's POST /v1/materialize request */
export interface MaterializeRequest {
  inputs: string[]
  transform: {
    executor: string
    params: Record<string, unknown>
  }
  env_hash: string
  mode?: 'artifact' | 'stream'
}

/** Maps to Strata's materialize response */
export interface MaterializeResponse {
  artifact_id: string
  version: number
  uri: string
  cache_hit: boolean
  provenance_hash: string
  state: string
  content_type?: ArtifactContentType
  schema?: { columns: string[] }
  row_count?: number
  stream_id?: string
}

/** DAG edge for visualization */
export interface DagEdge {
  /** Cell ID that defines the variable (snake_case matches backend) */
  from_cell_id: CellId
  /** Cell ID that references the variable */
  to_cell_id: CellId
  variable: string
}

/** Cascade plan — what needs to run before a target cell */
export interface CascadePlan {
  /** Target cell the user wants to run */
  targetCellId: CellId
  /** Cells that need to execute, in topological order */
  steps: CascadeStep[]
  /** Total estimated duration */
  estimatedMs: number
}

export interface CascadeStep {
  cellId: CellId
  cellName: string
  /** Whether this step can be skipped (already cached) */
  skip: boolean
  /** Why it needs to run */
  reason: 'stale' | 'missing' | 'target'
  /** Estimated duration */
  estimatedMs: number
}

/** Profiling summary for the entire notebook (v1.1) */
export interface ProfilingSummary {
  totalExecutionMs: number
  cacheHits: number
  cacheMisses: number
  cacheSavingsMs: number
  totalArtifactBytes: number
  cellProfiles: CellProfile[]
}

export interface CellProfile {
  cellId: CellId
  cellName: string
  status: CellStatus
  durationMs: number
  cacheHit: boolean
  artifactUri?: string
  executionCount: number
}

/** WebSocket message types: client → server */
export type WsClientMessageType =
  | 'cell_execute' // Run a cell (with cascade option)
  | 'cell_execute_cascade' // User confirmed cascade — execute the plan
  | 'cell_execute_force' // "Run this only" — execute with stale inputs
  | 'cell_cancel' // Cancel a running cell
  | 'cell_source_update' // Cell source changed (debounced)
  | 'notebook_run_all' // Run all cells (or just stale ones)
  | 'notebook_sync' // Reconnection — request full state
  | 'inspect_open' // Open inspect REPL for a cell
  | 'inspect_eval' // Evaluate expression in inspect REPL
  | 'inspect_close' // Close inspect REPL
  | 'impact_preview_request' // Request impact preview for a cell (v1.1)
  | 'profiling_request' // Request profiling summary (v1.1)
  | 'dependency_add' // Add a package dependency
  | 'dependency_remove' // Remove a package dependency
  | 'agent_cancel' // Cancel a running agent loop
  | 'agent_confirm_response' // User approved/declined a destructive tool call
  | 'variant_set_active' // Switch the active variant in a group
  | 'variant_add' // Add a new sibling variant to a group (clones active)

/** WebSocket message types: server → client */
export type WsServerMessageType =
  | 'cell_status' // Cell status changed (includes causality chain)
  | 'cell_output' // Cell produced output (artifact data for display)
  | 'cell_console' // Incremental console output (stdout/stderr)
  | 'cell_error' // Cell execution failed
  | 'cell_assertions' // Assertion results from cell execution
  | 'cell_iteration_progress' // Loop cell completed one iteration
  | 'dag_update' // Authoritative DAG from backend AST analysis
  | 'cascade_prompt' // "This cell needs N upstream cells to run first"
  | 'cascade_progress' // During cascade, reports which cell is running
  | 'impact_preview' // Run impact preview (upstream + downstream effects)
  | 'profiling_summary' // Notebook profiling summary (v1.1)
  | 'inspect_result' // Result of an inspect REPL evaluation
  | 'notebook_status' // Batch status update (e.g., after open or env change)
  | 'notebook_state' // Full state sync (reconnection)
  | 'dependency_changed' // Dependency added/removed — updated list
  | 'environment_job_started' // Background env job accepted and started
  | 'environment_job_progress' // Background env job emitted logs or phase changes
  | 'environment_job_finished' // Background env job completed or failed
  | 'error' // Protocol-level error (auth, not found, etc.)
  | 'agent_progress' // Agent loop progress event
  | 'agent_text_delta' // Streaming chunk of the agent's intermediate narrative
  | 'agent_confirm_request' // Agent wants to run a destructive tool — needs approval
  | 'agent_done' // Agent loop completed (success, error, or cancel)

export type WsMessageType = WsClientMessageType | WsServerMessageType

export interface WsMessage {
  type: WsMessageType
  cellId?: CellId
  /** Monotonic sequence number for ordering */
  seq: number
  /** Server timestamp (ISO 8601) */
  ts: string
  payload: unknown
}
