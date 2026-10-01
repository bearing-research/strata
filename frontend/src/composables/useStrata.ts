/** REST client for the Strata server. */

import { ref } from 'vue'
import type {
  CellOutput,
  ConnectionSpec,
  DependencyInfo,
  MaterializeRequest,
  MaterializeResponse,
  MountSpec,
} from '../types/notebook'

type UnknownObject = Record<string, unknown>

interface BackendCellPayload {
  id: string
  source?: string
  language?: 'python'
  order?: number
  worker?: string | null
  worker_override?: string | null
  timeout?: number | null
  timeout_override?: number | null
  env?: Record<string, string>
  env_overrides?: Record<string, string>
  mounts?: MountSpec[]
  mount_overrides?: MountSpec[]
  annotations?: UnknownObject
  defines?: string[]
  references?: string[]
  upstream_ids?: string[]
  downstream_ids?: string[]
  is_leaf?: boolean
}

interface NotebookSessionPayload {
  id: string
  name: string
  session_id: string
  path?: string
  cells?: BackendCellPayload[]
  workers?: UnknownObject[]
  mounts?: MountSpec[]
  environment?: UnknownObject
  environment_job?: UnknownObject | null
  environment_job_history?: UnknownObject[]
  dependencies?: DependencyInfo[]
  resolved_dependencies?: DependencyInfo[]
  dag?: UnknownObject
}

interface NotebookRenameResponse {
  name: string
}

interface NotebookDeleteResponse {
  name?: string
  path?: string
  session_id?: string
}

interface NotebookRuntimeConfigResponse {
  deployment_mode?: 'personal' | 'service'
  default_parent_path?: string
  available_python_versions?: string[]
  default_python_version?: string
  python_selection_fixed?: boolean
  registry_enabled?: boolean
  team_store_configured?: boolean
}

interface CellUpdateResponse {
  cell: BackendCellPayload
  dag: UnknownObject
  cells?: BackendCellPayload[]
}

// ---- Registry / dashboard ----

/** One artifact a cell published into the registry (GET …/artifacts). */
export interface PublishedArtifact {
  artifact_id: string
  version: number
  uri: string
  names: string[]
  tags: Record<string, string>
}

/** GET /v1/notebooks/{sid}/artifacts — keyed by cell id. */
interface NotebookArtifactsResponse {
  cells: Record<string, PublishedArtifact[]>
}

/** A protected-alias change awaiting approval (GET /v1/registry/pending). */
export interface PendingChange {
  name: string
  alias: string
  action: string
  artifact_id: string
  version: number
  requested_by: string | null
  requested_at: number
}

interface PendingChangesResponse {
  pending: PendingChange[]
}

/** One registry audit row (GET /v1/registry/audit). */
export interface AuditEntry {
  at: number
  action: string
  name: string | null
  alias: string | null
  artifact_id: string | null
  version: number | null
  actor: string | null
}

interface AuditResponse {
  entries: AuditEntry[]
}

/** Result of PUT …/aliases/{alias}: applied | pending | unchanged. */
export interface AliasMoveResult {
  status: string
  name?: string
  alias?: string
  detail?: string
  artifact_uri?: string
}

/** POST …/artifacts/{id}/v/{n}/promote: what landed in the team store. */
export interface PromotionResult {
  name: string
  artifact_uri: string
  copied: number
  alias: string | null
  alias_pending: boolean
  store: string
}

/** Flat lineage graph (GET …/lineage): nodes + edges (see lineageToTree). */
export interface LineageNode {
  uri: string
  artifact_id?: string | null
  version?: number | null
  type: string // 'artifact' | 'table'
  transform_ref?: string | null
  /** Who computed this step, when the store recorded an author. */
  principal?: string | null
  /** Interpreter + hardware that produced it, e.g. cpython-3.14-linux-x86_64. */
  build_env?: string
  /** How long the producing run took. */
  build_duration_ms?: number
  /** Digest of the environment it ran in; with build_env, the identity. */
  env_hash?: string
}
export interface LineageEdge {
  from_uri: string
  to_uri: string
  input_version?: string
}
export interface LineageGraph {
  artifact_uri: string
  nodes: LineageNode[]
  edges: LineageEdge[]
  direct_inputs?: string[]
}

/** One row of the registry names table (GET /v1/registry/summary). */
export interface RegistryName {
  name: string
  artifact_id: string
  version: number
  uri: string
  aliases: Record<string, number>
  tags: Record<string, string>
}

interface RegistrySummaryResponse {
  names: RegistryName[]
}

interface NotebookMutationResponse {
  cell?: BackendCellPayload
  cells?: BackendCellPayload[]
  mounts?: MountSpec[]
  workers?: UnknownObject[]
  configured_workers?: UnknownObject[]
  worker?: string | null
  timeout?: number | null
  env?: Record<string, string>
  definitions_editable?: boolean
  health_checked_at?: number | null
}

interface WorkerCatalogResponse {
  workers?: UnknownObject[]
  configured_workers?: UnknownObject[]
  definitions_editable?: boolean
  health_checked_at?: number | null
}

interface AdminNotebookWorkersResponse {
  configured_workers: UnknownObject[]
  health_checked_at?: number | null
}

interface DependencyListResponse {
  dependencies?: DependencyInfo[]
  resolved_dependencies?: DependencyInfo[]
}

interface EnvironmentResponse {
  environment?: UnknownObject
  environment_job?: UnknownObject | null
  environment_job_history?: UnknownObject[]
  dependencies?: DependencyInfo[]
  resolved_dependencies?: DependencyInfo[]
  cells?: BackendCellPayload[]
}

interface EnvironmentImportPreviewResponse {
  warnings?: string[]
  preview_dependencies?: DependencyInfo[]
  normalized_requirements?: string[]
  imported_count?: number
  additions?: DependencyInfo[]
  removals?: DependencyInfo[]
  unchanged?: DependencyInfo[]
}

interface NotebookSessionSummary {
  session_id: string
  name?: string
  path?: string
}

interface SessionListResponse {
  sessions?: NotebookSessionSummary[]
}

interface AddCellResponse extends BackendCellPayload {
  mounts?: MountSpec[]
  mount_overrides?: MountSpec[]
}

interface BackendWorkerPayload {
  name: string
  backend: 'local' | 'executor'
  runtime_id?: string | null
  config: Record<string, unknown>
}

interface BackendManagedWorkerPayload extends BackendWorkerPayload {
  enabled?: boolean
}

interface ApiErrorDetail {
  message?: string
  error?: string
}

interface ApiErrorPayload {
  detail?: string | ApiErrorDetail | null
  error?: string | ApiErrorDetail | null
}

type ErrorWithPayload = Error & { payload?: unknown }

function isUnknownObject(value: unknown): value is UnknownObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const DEFAULT_FETCH_TIMEOUT_MS = 30_000
const STREAM_FETCH_TIMEOUT_MS = 60_000

interface StrataFetchInit extends RequestInit {
  timeoutMs?: number
}

function resolveStrataBase(): string {
  const configured = import.meta.env.VITE_STRATA_URL
  if (configured) return configured
  if (typeof window !== 'undefined') return window.location.origin
  return 'http://localhost:8765'
}

const STRATA_BASE = resolveStrataBase()

const connected = ref(false)

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: StrataFetchInit = {},
): Promise<Response> {
  const { timeoutMs = DEFAULT_FETCH_TIMEOUT_MS, signal, ...rest } = init
  const controller = new AbortController()
  let timedOut = false
  const timeoutId = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  const abortFromSignal = () => controller.abort()
  if (signal) {
    if (signal.aborted) {
      abortFromSignal()
    } else {
      signal.addEventListener('abort', abortFromSignal, { once: true })
    }
  }

  try {
    return await fetch(input, { ...rest, signal: controller.signal })
  } catch (error) {
    if (timedOut && isAbortError(error)) {
      throw new Error(`Request timed out after ${timeoutMs}ms`)
    }
    throw error
  } finally {
    clearTimeout(timeoutId)
    if (signal) {
      signal.removeEventListener('abort', abortFromSignal)
    }
  }
}

async function readJson<T>(resp: Response): Promise<T> {
  return (await resp.json()) as T
}

// --- Mock execution (no server needed) ------------------------------------

function mockExecute(source: string): CellOutput {
  const lines = source.trim().split('\n')

  // If source looks like it assigns a list/dict, produce mock table
  if (source.includes('range(') || source.includes('[')) {
    const n = 10
    const columns = ['id', 'value', 'name']
    const rows = Array.from({ length: n }, (_, i) => ({
      id: i + 1,
      value: Math.round(Math.random() * 1000),
      name: `item_${i + 1}`,
    }))
    return { contentType: 'json/object', columns, rows, rowCount: n, cacheHit: false }
  }

  // If it references an upstream variable, pretend we got cached data
  if (source.includes('filter') || source.includes('query') || source.includes('select')) {
    const columns = ['id', 'value']
    const rows = Array.from({ length: 5 }, (_, i) => ({
      id: i * 10,
      value: Math.round(Math.random() * 500),
    }))
    return { contentType: 'json/object', columns, rows, rowCount: 5, cacheHit: true }
  }

  // Default: just show the code ran
  return {
    contentType: 'json/object',
    columns: ['result'],
    rows: [{ result: `Executed ${lines.length} line(s)` }],
    rowCount: 1,
    cacheHit: false,
  }
}

// --- Strata API calls -----------------------------------------------------

async function materialize(req: MaterializeRequest): Promise<MaterializeResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/materialize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!resp.ok) {
    throw new Error(`Strata error: ${resp.status} ${await resp.text()}`)
  }
  return readJson<MaterializeResponse>(resp)
}

async function fetchStream(streamId: string): Promise<ArrayBuffer> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/streams/${streamId}`, {
    timeoutMs: STREAM_FETCH_TIMEOUT_MS,
  })
  if (!resp.ok) throw new Error(`Stream error: ${resp.status}`)
  return resp.arrayBuffer()
}

async function throwApiError(resp: Response, fallback: string): Promise<never> {
  let payload: unknown = null
  let detail = ''
  try {
    payload = await resp.json()
    if (isUnknownObject(payload)) {
      const apiPayload = payload as ApiErrorPayload
      const rawDetail = apiPayload.detail ?? apiPayload.error
      if (rawDetail && typeof rawDetail === 'object') {
        detail = String(
          (rawDetail as ApiErrorDetail).message || (rawDetail as ApiErrorDetail).error || '',
        )
      } else {
        detail = String(rawDetail || '')
      }
    }
  } catch {
    try {
      detail = (await resp.text()).trim()
    } catch {
      detail = ''
    }
  }

  if (detail) {
    const error: ErrorWithPayload = new Error(detail)
    error.payload = payload
    throw error
  }
  const error: ErrorWithPayload = new Error(`${fallback}: ${resp.status}`)
  error.payload = payload
  throw error
}

// Probes the server but always returns mock output.
async function executeCell(source: string, _language: string): Promise<CellOutput> {
  try {
    const health = await fetchWithTimeout(`${STRATA_BASE}/health`, { timeoutMs: 500 })
    if (health.ok) {
      connected.value = true
    }
  } catch {
    connected.value = false
  }

  await new Promise((r) => setTimeout(r, 300 + Math.random() * 700))
  return mockExecute(source)
}

// --- Notebook API ---------------------------------------------------------

async function openNotebook(path: string): Promise<NotebookSessionPayload> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/open`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
    // Matches the backend's ``_renv_sync`` timeout: a first open with
    // ``renv.lock`` can spend that long compiling R packages from source.
    timeoutMs: 600_000,
  })
  if (!resp.ok) {
    throw new Error(`Failed to open notebook: ${resp.status}`)
  }
  return readJson<NotebookSessionPayload>(resp)
}

async function createNotebook(
  parentPath: string,
  name: string,
  pythonVersion?: string | null,
  starterCell = false,
): Promise<NotebookSessionPayload> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      parent_path: parentPath,
      name,
      ...(pythonVersion ? { python_version: pythonVersion } : {}),
      ...(starterCell ? { starter_cell: true } : {}),
    }),
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to create notebook')
  }
  return readJson<NotebookSessionPayload>(resp)
}

/**
 * Report from ``POST /v1/notebooks/import``; mirrors ``ImportResult`` in
 * ``strata.notebook.jupyter_import``.
 */
export interface ImportReport {
  markdown_cells: number
  code_cells: number
  suppressed_outputs: number
  skipped_cells: string[]
  translated_magics: string[]
  dropped_magics: string[]
  dropped_shells: string[]
  captured_deps: string[]
  warnings: string[]
  report_path: string | null
  report_text: string
}

export interface ImportNotebookResponse extends NotebookSessionPayload {
  import_report: ImportReport
}

async function importNotebook(
  file: File,
  options?: { name?: string; parentPath?: string },
): Promise<ImportNotebookResponse> {
  const form = new FormData()
  form.append('file', file, file.name)
  if (options?.name) form.append('name', options.name)
  if (options?.parentPath) form.append('parent_path', options.parentPath)
  // The server runs ``uv sync`` for the captured deps, slow on a cold cache.
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/import`, {
    method: 'POST',
    body: form,
    timeoutMs: 120_000,
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to import notebook')
  }
  return readJson<ImportNotebookResponse>(resp)
}

async function renameNotebook(notebookId: string, name: string): Promise<NotebookRenameResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/name`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to rename notebook')
  }
  return readJson<NotebookRenameResponse>(resp)
}

async function deleteNotebook(notebookId: string): Promise<NotebookDeleteResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}`, {
    method: 'DELETE',
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to delete notebook')
  }
  return readJson<NotebookDeleteResponse>(resp)
}

export interface DiscoveredNotebook {
  path: string
  name: string | null
  notebook_id: string | null
  updated_at: string | null
}

export interface DiscoverNotebooksResponse {
  root: string | null
  notebooks: DiscoveredNotebook[]
}

async function discoverNotebooks(): Promise<DiscoverNotebooksResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/discover`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to discover notebooks')
  }
  return readJson<DiscoverNotebooksResponse>(resp)
}

async function validateRecentNotebooks(paths: string[]): Promise<{ valid: string[] }> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/recents/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paths }),
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to validate recent notebooks')
  }
  return readJson<{ valid: string[] }>(resp)
}

async function deleteNotebookByPath(path: string): Promise<{ deleted: boolean; path: string }> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/delete-by-path`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to delete notebook')
  }
  return readJson<{ deleted: boolean; path: string }>(resp)
}

async function getNotebookRuntimeConfig(): Promise<NotebookRuntimeConfigResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/config`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to load notebook config')
  }
  return readJson<NotebookRuntimeConfigResponse>(resp)
}

// ---- Registry / dashboard ----
// Server registry routes; only the per-cell published list is notebook-scoped.
// Names go into the path unencoded: the route converters expect raw slashes.

async function getNotebookArtifacts(sessionId: string): Promise<NotebookArtifactsResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${sessionId}/artifacts`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to list published artifacts')
  }
  return readJson<NotebookArtifactsResponse>(resp)
}

async function setAlias(
  name: string,
  alias: string,
  artifactId: string,
  version: number,
): Promise<AliasMoveResult> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/names/${name}/aliases/${alias}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ artifact_id: artifactId, version }),
  })
  if (!resp.ok) {
    await throwApiError(resp, `Failed to set ${name}@${alias}`)
  }
  return readJson<AliasMoveResult>(resp)
}

async function promoteArtifact(
  sessionId: string,
  artifactId: string,
  version: number,
  body: { name: string; alias?: string; tags?: Record<string, string> },
): Promise<PromotionResult> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${sessionId}/artifacts/${encodeURIComponent(artifactId)}/v/${version}/promote`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, `Failed to promote ${body.name}`)
  }
  return readJson<PromotionResult>(resp)
}

async function getPendingChanges(): Promise<PendingChangesResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/registry/pending`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to load pending changes')
  }
  return readJson<PendingChangesResponse>(resp)
}

async function approvePending(name: string, alias: string): Promise<void> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/registry/pending/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, alias }),
  })
  if (!resp.ok) {
    await throwApiError(resp, `Failed to approve ${name}@${alias}`)
  }
}

async function rejectPending(name: string, alias: string): Promise<void> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/registry/pending/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, alias }),
  })
  if (!resp.ok) {
    await throwApiError(resp, `Failed to reject ${name}@${alias}`)
  }
}

async function getRegistrySummary(): Promise<RegistrySummaryResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/registry/summary`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to load registry summary')
  }
  return readJson<RegistrySummaryResponse>(resp)
}

async function getRegistryAudit(name?: string): Promise<AuditResponse> {
  const url = name
    ? `${STRATA_BASE}/v1/registry/audit?name=${encodeURIComponent(name)}`
    : `${STRATA_BASE}/v1/registry/audit`
  const resp = await fetchWithTimeout(url)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to load registry audit')
  }
  return readJson<AuditResponse>(resp)
}

async function getLineage(artifactId: string, version: number): Promise<LineageGraph> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/artifacts/${encodeURIComponent(artifactId)}/v/${version}/lineage`,
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to load lineage')
  }
  return readJson<LineageGraph>(resp)
}

export interface CellIterationInfo {
  iteration: number
  artifactUri: string
  artifactId: string
  version: number
  contentType: string
  byteSize: number
  rowCount: number | null
  createdAt: number | null
}

interface DataFilterSpec {
  col: string
  op: string
  value?: unknown
  value2?: unknown
}

interface CellDataPage {
  pageable: boolean
  columns: string[]
  rows: unknown[][]
  total: number
  offset: number
  limit: number
  sortBy: string | null
  sortDir: 'asc' | 'desc'
}

interface CellDataColumnSummary {
  name: string
  dtype: string
  nulls: number
  distinct: number
  min: unknown
  max: unknown
}

interface CellDataQuery {
  offset?: number
  limit?: number
  sortBy?: string | null
  sortDir?: 'asc' | 'desc'
  search?: string | null
  filters?: DataFilterSpec[]
}

function applyDataQueryParams(url: URL, opts: CellDataQuery): void {
  if (opts.sortBy) {
    url.searchParams.set('sort_by', opts.sortBy)
    url.searchParams.set('sort_dir', opts.sortDir ?? 'asc')
  }
  if (opts.search) {
    url.searchParams.set('search', opts.search)
  }
  if (opts.filters && opts.filters.length) {
    url.searchParams.set('filters', JSON.stringify(opts.filters))
  }
}

async function getCellData(
  notebookId: string,
  cellId: string,
  artifactUri: string,
  opts: CellDataQuery = {},
): Promise<CellDataPage> {
  const url = new URL(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/cells/${cellId}/data`,
    window.location.origin,
  )
  url.searchParams.set('artifact_uri', artifactUri)
  url.searchParams.set('offset', String(opts.offset ?? 0))
  url.searchParams.set('limit', String(opts.limit ?? 100))
  applyDataQueryParams(url, opts)
  const resp = await fetchWithTimeout(url.toString())
  if (!resp.ok) {
    throw new Error(`Failed to load table page: ${resp.status}`)
  }
  const raw = await readJson<Record<string, unknown>>(resp)
  return {
    pageable: Boolean(raw.pageable),
    columns: Array.isArray(raw.columns) ? raw.columns.map((c) => String(c)) : [],
    rows: Array.isArray(raw.rows) ? (raw.rows as unknown[][]) : [],
    total: Number(raw.total ?? 0),
    offset: Number(raw.offset ?? 0),
    limit: Number(raw.limit ?? 0),
    sortBy: raw.sort_by === null || raw.sort_by === undefined ? null : String(raw.sort_by),
    sortDir: raw.sort_dir === 'desc' ? 'desc' : 'asc',
  }
}

async function getCellDataSummary(
  notebookId: string,
  cellId: string,
  artifactUri: string,
): Promise<CellDataColumnSummary[]> {
  const url = new URL(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/cells/${cellId}/data/summary`,
    window.location.origin,
  )
  url.searchParams.set('artifact_uri', artifactUri)
  const resp = await fetchWithTimeout(url.toString())
  if (!resp.ok) {
    throw new Error(`Failed to load column summary: ${resp.status}`)
  }
  const raw = await readJson<Record<string, unknown>>(resp)
  const cols = Array.isArray(raw.columns) ? (raw.columns as Record<string, unknown>[]) : []
  return cols.map((c) => ({
    name: String(c.name),
    dtype: String(c.dtype),
    nulls: Number(c.nulls ?? 0),
    distinct: Number(c.distinct ?? 0),
    min: c.min ?? null,
    max: c.max ?? null,
  }))
}

function cellDataExportUrl(
  notebookId: string,
  cellId: string,
  artifactUri: string,
  fmt: 'csv' | 'parquet',
  opts: CellDataQuery = {},
): string {
  const url = new URL(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/cells/${cellId}/data/export`,
    window.location.origin,
  )
  url.searchParams.set('artifact_uri', artifactUri)
  url.searchParams.set('fmt', fmt)
  applyDataQueryParams(url, opts)
  return url.toString()
}

async function listCellIterations(
  notebookId: string,
  cellId: string,
  variable?: string,
): Promise<{ variable: string | null; iterations: CellIterationInfo[] }> {
  const url = new URL(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/cells/${cellId}/iterations`,
    window.location.origin,
  )
  if (variable) {
    url.searchParams.set('variable', variable)
  }
  const resp = await fetchWithTimeout(url.toString())
  if (!resp.ok) {
    throw new Error(`Failed to list iterations: ${resp.status}`)
  }
  const raw = await readJson<{
    variable: string | null
    iterations: Array<Record<string, unknown>>
  }>(resp)
  return {
    variable: raw.variable,
    iterations: raw.iterations.map((entry) => ({
      iteration: Number(entry.iteration),
      artifactUri: String(entry.artifact_uri),
      artifactId: String(entry.artifact_id),
      version: Number(entry.version),
      contentType: String(entry.content_type),
      byteSize: Number(entry.byte_size),
      rowCount:
        entry.row_count === null || entry.row_count === undefined ? null : Number(entry.row_count),
      createdAt:
        entry.created_at === null || entry.created_at === undefined
          ? null
          : Number(entry.created_at),
    })),
  }
}

async function updateCellSource(
  notebookId: string,
  cellId: string,
  source: string,
): Promise<CellUpdateResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/cells/${cellId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source }),
  })
  if (!resp.ok) {
    throw new Error(`Failed to update cell: ${resp.status}`)
  }
  return readJson<CellUpdateResponse>(resp)
}

async function addCell(
  notebookId: string,
  afterCellId?: string,
  language?: string,
): Promise<AddCellResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/cells`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      after_cell_id: afterCellId || null,
      ...(language ? { language } : {}),
    }),
  })
  if (!resp.ok) {
    throw new Error(`Failed to add cell: ${resp.status}`)
  }
  return readJson<AddCellResponse>(resp)
}

/**
 * Download the notebook's export. `GET /export` sends
 * `Content-Disposition: attachment`, so nothing is buffered in JS.
 */
function downloadExport(notebookId: string, format: 'markdown' | 'html', appView = false): void {
  const url =
    `${STRATA_BASE}/v1/notebooks/${notebookId}/export?fmt=${format}` +
    (appView ? '&app_view=1' : '')
  // window.open would flash a tab and window.location would navigate away.
  const link = document.createElement('a')
  link.href = url
  link.rel = 'noopener'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

async function removeCell(notebookId: string, cellId: string): Promise<unknown> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/cells/${cellId}`, {
    method: 'DELETE',
  })
  if (!resp.ok) {
    throw new Error(`Failed to remove cell: ${resp.status}`)
  }
  // Refreshed variant_groups + cells for the caller's variant cleanup.
  return resp.json().catch(() => null)
}

async function reorderCells(notebookId: string, cellIds: string[]): Promise<void> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/cells/reorder`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cell_ids: cellIds }),
  })
  if (!resp.ok) {
    throw new Error(`Failed to reorder cells: ${resp.status}`)
  }
}

async function updateNotebookMounts(
  notebookId: string,
  mounts: MountSpec[],
): Promise<NotebookMutationResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/mounts`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mounts }),
  })
  if (!resp.ok) {
    throw new Error(`Failed to update notebook mounts: ${resp.status}`)
  }
  return readJson<NotebookMutationResponse>(resp)
}

interface ConnectionSchemaColumn {
  name: string
  type: string
  nullable: boolean | null
}

interface ConnectionSchemaTable {
  catalog: string | null
  schema: string | null
  name: string
  columns: ConnectionSchemaColumn[]
}

interface ConnectionSchemaResponse {
  connection: string
  driver: string
  tables: ConnectionSchemaTable[]
}

async function getConnectionSchema(
  notebookId: string,
  connectionName: string,
): Promise<ConnectionSchemaResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/connections/${encodeURIComponent(
      connectionName,
    )}/schema`,
  )
  if (!resp.ok) {
    let detail = ''
    try {
      detail = ((await readJson<{ detail?: string }>(resp)).detail ?? '').toString()
    } catch {
      /* no JSON body: report the status alone */
    }
    throw new Error(`Schema fetch failed: ${resp.status}${detail ? ` — ${detail}` : ''}`)
  }
  return readJson<ConnectionSchemaResponse>(resp)
}

async function listNotebookConnections(notebookId: string): Promise<ConnectionSpec[]> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/connections`)
  if (!resp.ok) {
    throw new Error(`Failed to list notebook connections: ${resp.status}`)
  }
  const body = await readJson<{ connections?: ConnectionSpec[] }>(resp)
  return body.connections ?? []
}

async function updateNotebookConnections(
  notebookId: string,
  connections: ConnectionSpec[],
): Promise<{ connections: ConnectionSpec[]; malformed_connections: ConnectionSpec[] }> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/connections`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ connections }),
  })
  if (!resp.ok) {
    let detail = ''
    try {
      detail = ((await readJson<{ detail?: string }>(resp)).detail ?? '').toString()
    } catch {
      // no JSON body: report the status alone
    }
    throw new Error(
      `Failed to update notebook connections: ${resp.status}${detail ? ` — ${detail}` : ''}`,
    )
  }
  return readJson<{
    connections: ConnectionSpec[]
    malformed_connections: ConnectionSpec[]
  }>(resp)
}

async function updateNotebookWorker(
  notebookId: string,
  worker: string | null,
): Promise<NotebookMutationResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/worker`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ worker }),
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to update notebook worker')
  }
  return readJson<NotebookMutationResponse>(resp)
}

async function updateNotebookTimeout(
  notebookId: string,
  timeout: number | null,
): Promise<NotebookMutationResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/timeout`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ timeout }),
  })
  if (!resp.ok) {
    throw new Error(`Failed to update notebook timeout: ${resp.status}`)
  }
  return readJson<NotebookMutationResponse>(resp)
}

async function updateNotebookEnv(
  notebookId: string,
  env: Record<string, string>,
): Promise<NotebookMutationResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/env`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ env }),
  })
  if (!resp.ok) {
    throw new Error(`Failed to update notebook env: ${resp.status}`)
  }
  return readJson<NotebookMutationResponse>(resp)
}

async function refreshNotebookSecretManager(notebookId: string): Promise<NotebookMutationResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/secret-manager/refresh`,
    { method: 'POST' },
  )
  if (!resp.ok) {
    throw new Error(`Failed to refresh secret manager: ${resp.status}`)
  }
  return readJson<NotebookMutationResponse>(resp)
}

async function updateNotebookSecretManagerConfig(
  notebookId: string,
  config: {
    provider?: string | null
    project_id?: string | null
    environment?: string | null
    path?: string | null
    base_url?: string | null
  },
): Promise<NotebookMutationResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/secret-manager/config`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    },
  )
  if (!resp.ok) {
    throw new Error(`Failed to update secret manager config: ${resp.status}`)
  }
  return readJson<NotebookMutationResponse>(resp)
}

async function listWorkers(notebookId: string, refresh = false): Promise<WorkerCatalogResponse> {
  const params = refresh ? '?refresh=true' : ''
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/workers${params}`)
  if (!resp.ok) {
    throw new Error(`Failed to list workers: ${resp.status}`)
  }
  return readJson<WorkerCatalogResponse>(resp)
}

async function updateNotebookPythonVersion(
  notebookId: string,
  pythonVersion: string,
): Promise<{
  accepted: boolean
  reason?: string
}> {
  // 200 is a no-op (already at that version), 202 dispatches a job.
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/python-version`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ python_version: pythonVersion }),
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to update Python version')
  }
  return readJson<{ accepted: boolean; reason?: string }>(resp)
}

async function updateWorkers(
  notebookId: string,
  workers: BackendWorkerPayload[],
): Promise<WorkerCatalogResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/workers`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ workers }),
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to update workers')
  }
  return readJson<WorkerCatalogResponse>(resp)
}

async function listAdminNotebookWorkers(refresh = false): Promise<AdminNotebookWorkersResponse> {
  const params = refresh ? '?refresh=true' : ''
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/admin/notebook-workers${params}`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to list admin notebook workers')
  }
  return readJson<AdminNotebookWorkersResponse>(resp)
}

async function updateAdminNotebookWorkers(
  workers: BackendManagedWorkerPayload[],
): Promise<AdminNotebookWorkersResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/admin/notebook-workers`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ workers }),
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to update admin notebook workers')
  }
  return readJson<AdminNotebookWorkersResponse>(resp)
}

async function createAdminNotebookWorker(
  worker: BackendManagedWorkerPayload,
): Promise<AdminNotebookWorkersResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/admin/notebook-workers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(worker),
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to create admin notebook worker')
  }
  return readJson<AdminNotebookWorkersResponse>(resp)
}

async function replaceAdminNotebookWorker(
  workerName: string,
  worker: BackendManagedWorkerPayload,
): Promise<AdminNotebookWorkersResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/admin/notebook-workers/${encodeURIComponent(workerName)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(worker),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to replace admin notebook worker')
  }
  return readJson<AdminNotebookWorkersResponse>(resp)
}

async function patchAdminNotebookWorker(
  workerName: string,
  enabled: boolean,
): Promise<AdminNotebookWorkersResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/admin/notebook-workers/${encodeURIComponent(workerName)}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to update admin notebook worker')
  }
  return readJson<AdminNotebookWorkersResponse>(resp)
}

async function deleteAdminNotebookWorker(
  workerName: string,
): Promise<AdminNotebookWorkersResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/admin/notebook-workers/${encodeURIComponent(workerName)}`,
    {
      method: 'DELETE',
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to delete admin notebook worker')
  }
  return readJson<AdminNotebookWorkersResponse>(resp)
}

async function refreshAdminNotebookWorker(
  workerName: string,
): Promise<AdminNotebookWorkersResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/admin/notebook-workers/${encodeURIComponent(workerName)}/refresh`,
    {
      method: 'POST',
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to refresh admin notebook worker')
  }
  return readJson<AdminNotebookWorkersResponse>(resp)
}

// --- Dependency API -------------------------------------------------------

async function listDependencies(notebookId: string): Promise<DependencyListResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/dependencies`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to list dependencies')
  }
  return readJson<DependencyListResponse>(resp)
}

async function addDependency(notebookId: string, pkg: string): Promise<EnvironmentResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/environment/jobs`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'add', package: pkg }),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to add dependency')
  }
  return readJson<EnvironmentResponse>(resp)
}

// R env jobs use the same ``environment/jobs`` endpoint with an R ``action``.
async function initRenv(notebookId: string): Promise<EnvironmentResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/environment/jobs`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'r_init' }),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to initialise renv')
  }
  return readJson<EnvironmentResponse>(resp)
}

async function addRPackage(notebookId: string, pkg: string): Promise<EnvironmentResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/environment/jobs`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'r_add', package: pkg }),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to install R package')
  }
  return readJson<EnvironmentResponse>(resp)
}

async function removeDependency(notebookId: string, pkg: string): Promise<EnvironmentResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/environment/jobs`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'remove', package: pkg }),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to remove dependency')
  }
  return readJson<EnvironmentResponse>(resp)
}

async function getEnvironmentStatus(notebookId: string): Promise<EnvironmentResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/environment`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to load notebook environment')
  }
  return readJson<EnvironmentResponse>(resp)
}

interface RPackagesResponse {
  packages?: Array<{ name?: string; version?: string }>
  packages_status?: string
  packages_error?: string | null
}

async function getRPackages(notebookId: string): Promise<RPackagesResponse> {
  // Separate from getEnvironmentStatus because it spawns Rscript (~1-2s).
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/r-packages`, {
    // Above the backend's own 30s subprocess timeout.
    timeoutMs: 45_000,
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to load R packages')
  }
  return readJson<RPackagesResponse>(resp)
}

async function syncEnvironment(notebookId: string): Promise<EnvironmentResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/environment/jobs`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'sync' }),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to sync notebook environment')
  }
  return readJson<EnvironmentResponse>(resp)
}

async function exportRequirements(notebookId: string): Promise<string> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/environment/requirements.txt`,
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to export requirements.txt')
  }
  return resp.text()
}

async function importRequirements(
  notebookId: string,
  requirements: string,
): Promise<EnvironmentResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/environment/jobs`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'import', requirements }),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to import requirements.txt')
  }
  return readJson<EnvironmentResponse>(resp)
}

async function previewRequirementsImport(
  notebookId: string,
  requirements: string,
): Promise<EnvironmentImportPreviewResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/environment/requirements.txt/preview`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ requirements }),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to preview requirements.txt import')
  }
  return readJson<EnvironmentImportPreviewResponse>(resp)
}

async function importEnvironmentYaml(
  notebookId: string,
  environmentYaml: string,
): Promise<EnvironmentResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/environment/jobs`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'import', environment_yaml: environmentYaml }),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to import environment.yaml')
  }
  return readJson<EnvironmentResponse>(resp)
}

async function previewEnvironmentYamlImport(
  notebookId: string,
  environmentYaml: string,
): Promise<EnvironmentImportPreviewResponse> {
  const resp = await fetchWithTimeout(
    `${STRATA_BASE}/v1/notebooks/${notebookId}/environment/environment.yaml/preview`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ environment_yaml: environmentYaml }),
    },
  )
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to preview environment.yaml import')
  }
  return readJson<EnvironmentImportPreviewResponse>(resp)
}

// --- Session management ---------------------------------------------------

async function listSessions(): Promise<NotebookSessionSummary[]> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/sessions`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to list sessions')
  }
  const data = await readJson<SessionListResponse>(resp)
  return data.sessions ?? []
}

async function getSession(sessionId: string): Promise<NotebookSessionPayload> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/sessions/${sessionId}`)
  if (!resp.ok) {
    await throwApiError(resp, 'Session not found')
  }
  return readJson<NotebookSessionPayload>(resp)
}

// --- Logs -----------------------------------------------------------------

export interface LogEntry {
  cursor: number
  timestamp?: string
  level?: string
  logger?: string
  message?: string
  notebook_id?: string
  // Structured logging flattens arbitrary kwargs onto the record.
  [key: string]: unknown
}

export interface LogQuery {
  since?: number
  level?: string
  notebook?: string
  regex?: string
  limit?: number
}

export interface LogsResponse {
  entries: LogEntry[]
  cursor: number
}

async function getLogs(query: LogQuery = {}): Promise<LogsResponse> {
  const params = new URLSearchParams()
  if (query.since !== undefined) params.set('since', String(query.since))
  if (query.level) params.set('level', query.level)
  if (query.notebook) params.set('notebook', query.notebook)
  if (query.regex) params.set('regex', query.regex)
  if (query.limit !== undefined) params.set('limit', String(query.limit))
  const qs = params.toString()
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/logs${qs ? `?${qs}` : ''}`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to fetch logs')
  }
  const data = await readJson<Partial<LogsResponse>>(resp)
  return { entries: data.entries ?? [], cursor: data.cursor ?? 0 }
}

// --- Artifacts ------------------------------------------------------------

export interface ArtifactStats {
  total_versions: number
  ready_versions: number
  building_versions: number
  failed_versions: number
  total_bytes: number
  total_rows: number
  name_count: number
}

export interface ArtifactRow {
  artifact_uri: string
  artifact_id: string
  version: number
  state: string
  row_count: number | null
  byte_size: number | null
  created_at: number | null
}

export interface ArtifactQuery {
  limit?: number
  offset?: number
  state?: string
  namePrefix?: string
  since?: number
  sort?: 'created_at' | 'byte_size' | 'row_count'
  order?: 'asc' | 'desc'
}

export interface ArtifactListResponse {
  artifacts: ArtifactRow[]
  limit: number
  offset: number
}

async function getArtifactStats(): Promise<ArtifactStats> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/artifacts/stats`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to fetch artifact stats')
  }
  return readJson<ArtifactStats>(resp)
}

async function getArtifacts(query: ArtifactQuery = {}): Promise<ArtifactListResponse> {
  const params = new URLSearchParams()
  if (query.limit !== undefined) params.set('limit', String(query.limit))
  if (query.offset !== undefined) params.set('offset', String(query.offset))
  if (query.state) params.set('state', query.state)
  if (query.namePrefix) params.set('name_prefix', query.namePrefix)
  if (query.since !== undefined) params.set('since', String(query.since))
  if (query.sort) params.set('sort', query.sort)
  if (query.order) params.set('order', query.order)
  const qs = params.toString()
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/artifacts${qs ? `?${qs}` : ''}`)
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to fetch artifacts')
  }
  const data = await readJson<Partial<ArtifactListResponse>>(resp)
  return {
    artifacts: data.artifacts ?? [],
    limit: data.limit ?? query.limit ?? 100,
    offset: data.offset ?? query.offset ?? 0,
  }
}

// --- Public API -----------------------------------------------------------

export function useStrata() {
  return {
    connected,
    executeCell,
    materialize,
    fetchStream,
    openNotebook,
    discoverNotebooks,
    createNotebook,
    importNotebook,
    renameNotebook,
    deleteNotebook,
    deleteNotebookByPath,
    validateRecentNotebooks,
    getNotebookRuntimeConfig,
    getNotebookArtifacts,
    getRegistrySummary,
    setAlias,
    promoteArtifact,
    getPendingChanges,
    approvePending,
    rejectPending,
    getRegistryAudit,
    getLineage,
    updateCellSource,
    addCell,
    removeCell,
    downloadExport,
    reorderCells,
    getCellData,
    getCellDataSummary,
    cellDataExportUrl,
    listCellIterations,
    updateNotebookMounts,
    listNotebookConnections,
    updateNotebookConnections,
    getConnectionSchema,
    updateNotebookWorker,
    updateNotebookTimeout,
    updateNotebookPythonVersion,
    updateNotebookEnv,
    refreshNotebookSecretManager,
    updateNotebookSecretManagerConfig,
    listWorkers,
    updateWorkers,
    listAdminNotebookWorkers,
    updateAdminNotebookWorkers,
    createAdminNotebookWorker,
    replaceAdminNotebookWorker,
    patchAdminNotebookWorker,
    deleteAdminNotebookWorker,
    refreshAdminNotebookWorker,
    listDependencies,
    addDependency,
    removeDependency,
    getEnvironmentStatus,
    getRPackages,
    initRenv,
    addRPackage,
    syncEnvironment,
    exportRequirements,
    previewRequirementsImport,
    importRequirements,
    previewEnvironmentYamlImport,
    importEnvironmentYaml,
    listSessions,
    getSession,
    getLogs,
    getArtifactStats,
    getArtifacts,
  }
}
