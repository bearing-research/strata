/**
 * Communication layer with the Strata server.
 *
 * Day-one approach: REST calls to /v1/materialize + a mock fallback
 * so the UI works even without a running server.
 */

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
}

interface CellUpdateResponse {
  cell: BackendCellPayload
  dag: UnknownObject
  cells?: BackendCellPayload[]
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

// ---------------------------------------------------------------------------
// Mock execution — lets us demo the UI without a live server
// ---------------------------------------------------------------------------

function mockExecute(source: string): CellOutput {
  // Simulate a Python cell that produces tabular data
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

// ---------------------------------------------------------------------------
// Real Strata API calls
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Execute a cell — try real server, fall back to mock
// ---------------------------------------------------------------------------

async function executeCell(source: string, _language: string): Promise<CellOutput> {
  // For day-one: always use mock. When server is running, swap to real.
  try {
    const health = await fetchWithTimeout(`${STRATA_BASE}/health`, { timeoutMs: 500 })
    if (health.ok) {
      connected.value = true
      // TODO: Wire to real materialize call once notebook backend is ready
      // For now, even with server up, use mock since we don't have notebook endpoints yet
    }
  } catch {
    connected.value = false
  }

  // Simulate async work
  await new Promise((r) => setTimeout(r, 300 + Math.random() * 700))
  return mockExecute(source)
}

// ---------------------------------------------------------------------------
// Notebook API functions
// ---------------------------------------------------------------------------

async function openNotebook(path: string): Promise<NotebookSessionPayload> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/open`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
    // 10 minutes — matches the backend's ``_renv_sync`` timeout
    // (``writer.py:_renv_sync``). A cold first-open of a notebook
    // with ``renv.lock`` can spend that long compiling R packages
    // from source. Same-or-cached-lockfile reopens short-circuit
    // inside ``ensure_renv_synced`` in milliseconds, so this
    // generous cap only matters on the first restore. Python-only
    // notebooks' ``uv sync`` finishes well under a minute even on
    // cold start, so this isn't a regression for them.
    //
    // Background-restore + WS-pushed progress is the architectural
    // answer for the slow first-open UX; this timeout bump is the
    // minimum guard that "open didn't time out the wire" doesn't
    // shadow a real, working restore in progress.
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
 * Outcome of importing a Jupyter ``.ipynb`` file via
 * ``POST /v1/notebooks/import``. Mirrors the ``ImportResult`` field
 * shapes from ``strata.notebook.jupyter_import`` so the UI can
 * display the conversion report inline.
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
  // The server runs ``uv sync`` after the conversion to install the
  // captured deps, so this can take a while on a cold cache. Match
  // the openNotebook timeout (120s) which has the same characteristic.
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
 * Trigger a browser download of the notebook's export. Hits the
 * server-side `GET /export?fmt=...` endpoint, which returns the
 * rendered content with `Content-Disposition: attachment` so the
 * browser handles the save dialog. No file is buffered in JS.
 */
function downloadExport(notebookId: string, format: 'markdown' | 'html'): void {
  const url = `${STRATA_BASE}/v1/notebooks/${notebookId}/export?fmt=${format}`
  // A hidden anchor click is the canonical "start a download" trick;
  // window.open would briefly flash a tab, window.location would
  // navigate away. The anchor stays invisible.
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
  // Body carries refreshed variant_groups + cells for variant-aware
  // delete cleanup. Caller picks them up; returning a parsed value
  // instead of void is backwards compatible.
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
      /* ignore */
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
      // ignore
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
  // 200 (no-op when already at requested version) or 202 (job dispatched)
  // both indicate the request was understood. Errors throw via
  // throwApiError so callers see 400/404/409 with the server's detail.
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

// ---------------------------------------------------------------------------
// Dependency API
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// LLM Assistant
// ---------------------------------------------------------------------------

interface LlmStatusResponse {
  available: boolean
  model: string | null
  provider: string | null
}

export interface LlmChatTurn {
  role: 'user' | 'assistant'
  content: string
}

export interface LlmStreamDelta {
  type: 'delta'
  text: string
}

export interface LlmStreamDone {
  type: 'done'
  model: string | null
  tokens: { input: number; output: number }
}

export interface LlmStreamError {
  type: 'error'
  message: string
}

export type LlmStreamEvent = LlmStreamDelta | LlmStreamDone | LlmStreamError

async function getLlmStatus(notebookId: string): Promise<LlmStatusResponse> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/ai/status`)
  if (!resp.ok) {
    return { available: false, model: null, provider: null }
  }
  return readJson<LlmStatusResponse>(resp)
}

async function getLlmModels(
  notebookId: string,
): Promise<{ models: string[]; current: string | null; provider: string | null }> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/ai/models`, {
    timeoutMs: 15_000,
  })
  if (!resp.ok) {
    return { models: [], current: null, provider: null }
  }
  return readJson<{ models: string[]; current: string | null; provider: string | null }>(resp)
}

async function updateLlmModel(notebookId: string, model: string): Promise<{ model: string }> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/ai/model`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Failed to update model')
  }
  return readJson<{ model: string }>(resp)
}

/**
 * Stream a chat completion from /ai/stream via SSE.
 *
 * Yields incremental deltas, a final `done` event, or a single `error`
 * event. The caller is responsible for appending deltas to the assistant
 * message in the UI.
 */
async function* llmChatStream(
  notebookId: string,
  message: string,
  history: LlmChatTurn[],
  cellId?: string,
  signal?: AbortSignal,
): AsyncGenerator<LlmStreamEvent, void, void> {
  const resp = await fetch(`${STRATA_BASE}/v1/notebooks/${notebookId}/ai/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      message,
      history,
      cell_id: cellId ?? null,
    }),
    signal,
  })

  if (!resp.ok) {
    let detail = `LLM stream failed (${resp.status})`
    try {
      const body = await resp.json()
      if (body?.detail) detail = body.detail
    } catch {
      // ignore
    }
    yield { type: 'error', message: detail }
    return
  }

  if (!resp.body) {
    yield { type: 'error', message: 'LLM stream returned no body' }
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      let sepIdx: number
      while ((sepIdx = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, sepIdx)
        buffer = buffer.slice(sepIdx + 2)
        const parsed = parseSseEvent(rawEvent)
        if (parsed) yield parsed
      }
    }
  } finally {
    try {
      reader.releaseLock()
    } catch {
      // ignore
    }
  }
}

function parseSseEvent(raw: string): LlmStreamEvent | null {
  let eventName = 'message'
  const dataLines: string[] = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim())
    }
  }
  if (dataLines.length === 0) return null
  const dataStr = dataLines.join('\n')
  let data: any
  try {
    data = JSON.parse(dataStr)
  } catch {
    return null
  }
  if (eventName === 'delta' && typeof data?.text === 'string') {
    return { type: 'delta', text: data.text }
  }
  if (eventName === 'done') {
    return {
      type: 'done',
      model: data?.model ?? null,
      tokens: {
        input: data?.tokens?.input ?? 0,
        output: data?.tokens?.output ?? 0,
      },
    }
  }
  if (eventName === 'error') {
    return { type: 'error', message: data?.message ?? 'LLM stream error' }
  }
  return null
}

async function agentRun(
  notebookId: string,
  message: string,
  opts: { autoApprove?: boolean } = {},
): Promise<any> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/ai/agent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, auto_approve: opts.autoApprove ?? false }),
    timeoutMs: 30_000, // Just needs to start the background task
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Agent run failed')
  }
  return resp.json()
}

async function agentReset(notebookId: string): Promise<void> {
  const resp = await fetchWithTimeout(`${STRATA_BASE}/v1/notebooks/${notebookId}/ai/agent/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    timeoutMs: 5_000,
  })
  if (!resp.ok) {
    await throwApiError(resp, 'Agent reset failed')
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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
    updateCellSource,
    addCell,
    removeCell,
    downloadExport,
    reorderCells,
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
    syncEnvironment,
    exportRequirements,
    previewRequirementsImport,
    importRequirements,
    previewEnvironmentYamlImport,
    importEnvironmentYaml,
    listSessions,
    getSession,
    getLlmStatus,
    getLlmModels,
    updateLlmModel,
    llmChatStream,
    agentRun,
    agentReset,
  }
}
