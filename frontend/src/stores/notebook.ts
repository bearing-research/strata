import { reactive, computed, ref } from 'vue'
import type {
  Cell,
  CellId,
  CellOutput,
  CellStatus,
  CellTestResult,
  CellTestStatus,
  ConnectionSpec,
  DagEdge,
  DependencyInfo,
  EnvironmentImportPreview,
  LoopProgress,
  VariantProgress,
  Notebook,
  WsMessage,
  ImpactPreview,
  ProfilingSummary,
  MountSpec,
  CellAnnotations,
  EnvironmentActionSummary,
  EnvironmentJobAction,
  EnvironmentOperation,
  EditableWorkerSpec,
  ManagedWorkerSpec,
  NotebookEnvironment,
  NotebookRuntimeConfig,
  RNotebookEnvironment,
  VariantGroup,
  VariantMember,
  WorkerCatalogEntry,
  WorkerHealth,
  WorkerHealthHistoryEntry,
  WorkerSpec,
  WidgetSpec,
} from '../types/notebook'
import { useStrata } from '../composables/useStrata'
import type {
  AliasMoveResult,
  AuditEntry,
  PendingChange,
  PublishedArtifact,
  RegistryName,
} from '../composables/useStrata'
import { useWebSocket } from '../composables/useWebSocket'
import { flattenLineage, lineageToTree, type LineageTreeNode } from '../utils/lineage'
import { markNotebookPerf, measureNotebookPerf } from '../utils/perf'
import { consumePrefetchedNotebookSession } from '../utils/notebookSessionPrefetch'
import {
  applyWorkerHealth,
  effectiveWorkerNameForCell,
  isRemoteExecutorLikelyUnreachable,
} from '../utils/notebookWorkers'

let nextOrder = 0
const FALLBACK_NOTEBOOK_PARENT_PATH = '/tmp/strata-notebooks'

// ---------------------------------------------------------------------------
// Connection state — visible to the UI
// ---------------------------------------------------------------------------
const connected = ref(false)
const connectError = ref<string | null>(null)

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

const notebook = reactive<Notebook>({
  id: '',
  name: 'Untitled Notebook',
  worker: null,
  timeout: null,
  env: {},
  envSources: {},
  envFetchError: null,
  envFetchedAt: null,
  secretManagerConfig: {},
  workers: [],
  mounts: [],
  connections: [],
  cells: [],
  variantGroups: [],
  environment: {
    pythonVersion: '',
    requestedPythonVersion: '',
    runtimePythonVersion: '',
    lockfileHash: '',
    packageCount: 0,
    declaredPackageCount: 0,
    resolvedPackageCount: 0,
    syncState: 'unknown',
    syncError: null,
    syncNotice: null,
    lastSyncedAt: null,
    lastSyncDurationMs: null,
    hasLockfile: false,
    venvPython: null,
    interpreterSource: 'unknown',
  },
  rEnvironment: {
    hasLockfile: false,
    currentLockHash: '',
    lockHash: '',
    rVersion: '',
    systemRVersion: '',
    lastSyncedAt: 0,
    syncState: 'absent',
    syncError: null,
    // Default to ``unknown`` so the panel's first render renders
    // a "loading" indicator rather than a misleading "no packages
    // installed" — the env panel fetches the list on mount.
    packages: [],
    packagesStatus: 'unknown',
    packagesError: null,
  },
  createdAt: Date.now(),
  updatedAt: Date.now(),
})

// --- Derived state ---------------------------------------------------------

const cellMap = computed(() => {
  const m = new Map<CellId, Cell>()
  for (const c of notebook.cells) m.set(c.id, c)
  return m
})

const orderedCells = computed(() => [...notebook.cells].sort((a, b) => a.order - b.order))

/**
 * DAG edges authoritatively supplied by the backend.
 *
 * Recomputing locally from defines/references used a "last define wins"
 * map that dropped legitimate upstream edges when a cell both defined
 * and referenced a variable (e.g. a loop cell whose body rebinds the
 * carry). The backend DAG builder walks cells in order and resolves
 * the correct producer for each reference, so we use its output
 * directly and keep the client side for layout only.
 */
const backendDagEdges = ref<DagEdge[]>([])

const dagEdges = computed<DagEdge[]>(() => backendDagEdges.value)

// --- Helper: sessionId accessor -------------------------------------------

function sessionId(): string | undefined {
  return (notebook as any).sessionId
}

// --- Cell mutations (always go through backend when connected) -------------

async function addCell(afterId?: CellId, language?: string) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  try {
    const data = await strata.addCell(sid, afterId, language)
    const newCell: Cell = {
      id: data.id,
      source: data.source || '',
      language: data.language || 'python',
      order: data.order ?? nextOrder++,
      status: 'idle',
      worker: data.worker ?? notebook.worker ?? null,
      workerOverride: data.worker_override ?? null,
      timeout: data.timeout ?? notebook.timeout ?? null,
      timeoutOverride: data.timeout_override ?? null,
      env: parseEnvMap(data.env),
      envOverrides: parseEnvMap(data.env_overrides),
      mounts: Array.isArray(data.mounts) ? data.mounts.map(parseMountSpec) : [...notebook.mounts],
      mountOverrides: Array.isArray(data.mount_overrides)
        ? data.mount_overrides.map(parseMountSpec)
        : [],
      annotations: parseBackendAnnotations(data.annotations) || {
        worker: null,
        timeout: null,
        env: {},
        mounts: [],
      },
      upstreamIds: [],
      downstreamIds: [],
      defines: [],
      references: [],
      inputs: [],
      isLeaf: false,
    }
    const idx = afterId
      ? notebook.cells.findIndex((c) => c.id === afterId) + 1
      : notebook.cells.length
    notebook.cells.splice(idx, 0, newCell)
    notebook.updatedAt = Date.now()
  } catch (err) {
    console.error('Failed to add cell:', err)
  }
}

function removeCell(id: CellId) {
  if (notebook.cells.length <= 1) return
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  strata
    .removeCell(sid, id)
    .then((response: any) => {
      const idx = notebook.cells.findIndex((c) => c.id === id)
      if (idx >= 0) {
        notebook.cells.splice(idx, 1)
        notebook.updatedAt = Date.now()
      }
      // Variant cleanup: when the deleted cell was part of a group, the
      // backend may have promoted a sibling to active (its variantActive
      // flips true) or dissolved the group entirely. Sync from the
      // response so the tab strip + variant flags + DAG edges reflect
      // the new state without waiting for a manual reload.
      if (response && typeof response === 'object') {
        if (Array.isArray(response.variant_groups)) {
          notebook.variantGroups = parseBackendVariantGroups(response.variant_groups)
          syncCellVariantFlagsFromGroups()
        }
        if (Array.isArray(response.cells)) {
          for (const sc of response.cells) {
            const cell = cellMap.value.get(sc.id as CellId)
            if (!cell) continue
            if (Array.isArray(sc.upstream_ids)) cell.upstreamIds = sc.upstream_ids
            if (Array.isArray(sc.downstream_ids)) cell.downstreamIds = sc.downstream_ids
          }
        }
        if (response.dag && typeof response.dag === 'object') {
          applyBackendDag(response.dag)
        }
      }
    })
    .catch((err) => {
      console.error('Failed to remove cell:', err)
    })
}

// --- Local-first source editing ------------------------------------------------
// Editing is instant and local. Backend sync happens asynchronously via
// WebSocket — the editor never blocks on a round-trip during typing.
// The backend broadcasts dag_update + cell_status when analysis finishes.

const dirtyCells = new Set<CellId>()
let idleFlushTimer: ReturnType<typeof setTimeout> | null = null
const IDLE_FLUSH_MS = 2000

function updateSource(id: CellId, source: string) {
  const cell = cellMap.value.get(id)
  if (!cell) return
  cell.source = source

  // A source edit invalidates the last test result (the cell the tests ran
  // against has changed). Cheap, edit-driven staleness — see runCellTests.
  if (cell.testResult && !cell.testResult.stale) {
    cell.testResult = { ...cell.testResult, stale: true }
  }

  // Mark dirty for async backend sync. No local regex extraction —
  // defines/references/DAG update from the backend after flush to
  // avoid noisy partial-word references mid-typing.
  dirtyCells.add(id)
  scheduleIdleFlush()

  notebook.updatedAt = Date.now()
}

function scheduleIdleFlush() {
  if (idleFlushTimer) clearTimeout(idleFlushTimer)
  idleFlushTimer = setTimeout(() => {
    idleFlushTimer = null
    flushDirtyCells()
  }, IDLE_FLUSH_MS)
}

function flushDirtyCells() {
  if (idleFlushTimer) {
    clearTimeout(idleFlushTimer)
    idleFlushTimer = null
  }
  if (dirtyCells.size === 0) return
  for (const cellId of dirtyCells) {
    const cell = cellMap.value.get(cellId)
    if (!cell) continue
    updateSourceWebSocket(cellId, cell.source)
  }
  dirtyCells.clear()
}

function flushCellSource(cellId: CellId) {
  if (!dirtyCells.has(cellId)) return
  const cell = cellMap.value.get(cellId)
  if (!cell) return
  updateSourceWebSocket(cellId, cell.source)
  dirtyCells.delete(cellId)
}

function setCellStatus(id: CellId, status: CellStatus) {
  const cell = cellMap.value.get(id)
  if (cell) cell.status = status
}

function setCellOutput(id: CellId, output: CellOutput, displayOutputs?: CellOutput[]) {
  const cell = cellMap.value.get(id)
  if (cell) {
    cell.displayOutputs = displayOutputs ?? (output.error ? [] : [output])
    cell.output = output
    cell.status = output.error ? 'error' : 'ready'
    cell.lastRunAt = Date.now()
  }
}

function parseBackendCausality(raw: any): Cell['causality'] | undefined {
  if (!raw) return undefined
  return {
    reason: raw.reason,
    details: (raw.details || []).map((d: Record<string, any>) => ({
      type: d.type,
      cellId: d.cell_id,
      cellName: d.cell_name,
      fromVersion: d.from_version,
      toVersion: d.to_version,
      package: d.package,
      fromPackageVersion: d.from_package_version,
      toPackageVersion: d.to_package_version,
    })),
  }
}

function supportsStalenessDetail(status: CellStatus | undefined): boolean {
  return status === 'idle' || status === 'stale'
}

function parseMountSpec(raw: any): MountSpec {
  return {
    name: raw.name,
    uri: raw.uri,
    mode: raw.mode || 'ro',
    pin: raw.pin ?? null,
  }
}

function parseBackendEnvironment(raw: any): NotebookEnvironment {
  const declaredPackageCount = Number(
    raw?.declared_package_count ??
      raw?.declaredPackageCount ??
      raw?.package_count ??
      raw?.packageCount ??
      0,
  )
  const runtimePythonVersion = String(
    raw?.runtime_python_version ??
      raw?.runtimePythonVersion ??
      raw?.python_version ??
      raw?.pythonVersion ??
      '',
  )
  const requestedPythonVersion = String(
    raw?.requested_python_version ?? raw?.requestedPythonVersion ?? runtimePythonVersion,
  )
  return {
    pythonVersion: runtimePythonVersion,
    requestedPythonVersion,
    runtimePythonVersion,
    lockfileHash: String(raw?.lockfile_hash ?? raw?.lockfileHash ?? ''),
    packageCount: declaredPackageCount,
    declaredPackageCount,
    resolvedPackageCount: Number(raw?.resolved_package_count ?? raw?.resolvedPackageCount ?? 0),
    syncState: ['ready', 'fallback', 'failed', 'pending', 'unknown'].includes(raw?.sync_state)
      ? raw.sync_state
      : 'unknown',
    syncError: typeof raw?.sync_error === 'string' ? raw.sync_error : null,
    syncNotice: typeof raw?.sync_notice === 'string' ? raw.sync_notice : null,
    lastSyncedAt:
      typeof raw?.last_synced_at === 'number'
        ? raw.last_synced_at
        : typeof raw?.lastSyncedAt === 'number'
          ? raw.lastSyncedAt
          : null,
    lastSyncDurationMs:
      typeof raw?.last_sync_duration_ms === 'number'
        ? raw.last_sync_duration_ms
        : typeof raw?.lastSyncDurationMs === 'number'
          ? raw.lastSyncDurationMs
          : null,
    hasLockfile: raw?.has_lockfile === true || raw?.hasLockfile === true,
    venvPython:
      typeof raw?.venv_python === 'string'
        ? raw.venv_python
        : typeof raw?.venvPython === 'string'
          ? raw.venvPython
          : null,
    interpreterSource:
      raw?.interpreter_source === 'venv' || raw?.interpreterSource === 'venv'
        ? 'venv'
        : raw?.interpreter_source === 'path' || raw?.interpreterSource === 'path'
          ? 'path'
          : 'unknown',
  }
}

function parseDependencyInfo(raw: any): DependencyInfo {
  return {
    name: String(raw?.name || ''),
    version: String(raw?.version || ''),
    specifier: String(raw?.specifier || ''),
  }
}

function parseEnvironmentOperation(raw: any): EnvironmentOperation | null {
  if (!raw || typeof raw !== 'object') return null
  // Whitelist mirrors the backend's ``EnvironmentJobRequest`` valid_actions.
  // Missing entries here previously caused ``r_init`` / ``r_add`` jobs to
  // parse as ``null`` — the env-operation panel flashed once on the
  // optimistic local ``beginEnvironmentOperation`` call, then got wiped
  // by the WS ``environment_job_started`` round-trip.
  const action =
    raw.action === 'add' ||
    raw.action === 'remove' ||
    raw.action === 'sync' ||
    raw.action === 'import' ||
    raw.action === 'r_init' ||
    raw.action === 'r_add' ||
    raw.action === 'change_python'
      ? raw.action
      : null
  if (!action) return null
  const status = raw.status === 'completed' || raw.status === 'failed' ? raw.status : 'running'
  return {
    id: typeof raw.id === 'string' ? raw.id : `${action}-${Date.now()}`,
    action,
    status,
    packageName: typeof raw.package === 'string' ? raw.package : null,
    phase: typeof raw.phase === 'string' ? raw.phase : null,
    command: typeof raw.command === 'string' ? raw.command : '',
    durationMs: typeof raw.duration_ms === 'number' ? raw.duration_ms : null,
    stdout: typeof raw.stdout === 'string' ? raw.stdout : '',
    stderr: typeof raw.stderr === 'string' ? raw.stderr : '',
    stdoutTruncated: raw.stdout_truncated === true,
    stderrTruncated: raw.stderr_truncated === true,
    startedAt: typeof raw.started_at === 'number' ? raw.started_at : Date.now(),
    finishedAt: typeof raw.finished_at === 'number' ? raw.finished_at : null,
    lockfileChanged: raw.lockfile_changed === true,
    staleCellCount: typeof raw.stale_cell_count === 'number' ? raw.stale_cell_count : 0,
    staleCellIds: Array.isArray(raw.stale_cell_ids)
      ? raw.stale_cell_ids
          .map((value: unknown) => String(value || '').trim())
          .filter((value: string) => value.length > 0)
      : [],
    error: typeof raw.error === 'string' ? raw.error : null,
  }
}

function syncEnvironmentOperationFromBackend(raw: any) {
  const parsed = parseEnvironmentOperation(raw)
  if (!parsed) {
    environmentOperation.value = null
    return
  }
  environmentOperation.value = parsed
  const isRunning = parsed.status === 'running'
  dependencyLoading.value = isRunning
  environmentLoading.value = isRunning
}

function syncEnvironmentJobHistoryFromBackend(raw: any) {
  environmentJobHistory.value = Array.isArray(raw)
    ? raw
        .map((entry: any) => parseEnvironmentOperation(entry))
        .filter(
          (entry: EnvironmentOperation | null): entry is EnvironmentOperation => entry !== null,
        )
    : []
}

function syncResolvedDependenciesFromBackend(raw: any) {
  resolvedDependencies.value = Array.isArray(raw)
    ? raw.map((dep: any) => parseDependencyInfo(dep)).filter((dep) => dep.name)
    : []
}

function parseBackendREnvironment(
  raw: any,
): Omit<RNotebookEnvironment, 'packages' | 'packagesStatus' | 'packagesError'> {
  // The env-state payload omits the package list — that's served
  // by a separate ``GET /r-packages`` route so notebook open /
  // state sync / env refresh paths don't pay a synchronous Rscript
  // spawn. Callers reuse the store's existing
  // ``packages``/``packagesStatus``/``packagesError`` fields and
  // overwrite them via ``fetchRPackagesAction``.
  const syncStateRaw = String(raw?.sync_state ?? raw?.syncState ?? 'absent')
  const syncState = (
    ['absent', 'never', 'ok', 'outdated', 'failed'].includes(syncStateRaw) ? syncStateRaw : 'absent'
  ) as RNotebookEnvironment['syncState']
  const errRaw = raw?.sync_error ?? raw?.syncError
  return {
    hasLockfile: raw?.has_lockfile === true || raw?.hasLockfile === true,
    currentLockHash: String(raw?.current_lock_hash ?? raw?.currentLockHash ?? ''),
    lockHash: String(raw?.lock_hash ?? raw?.lockHash ?? ''),
    rVersion: String(raw?.r_version ?? raw?.rVersion ?? ''),
    systemRVersion: String(raw?.system_r_version ?? raw?.systemRVersion ?? ''),
    lastSyncedAt: Number(raw?.last_synced_at ?? raw?.lastSyncedAt ?? 0),
    syncState,
    syncError: typeof errRaw === 'string' && errRaw.length > 0 ? errRaw : null,
  }
}

function syncNotebookREnvironmentFromBackend(raw: any) {
  // Overlay the parsed env-state fields on top of the existing
  // packages/packagesStatus/packagesError fields so a state-sync
  // doesn't clobber a previously-fetched package listing.
  notebook.rEnvironment = {
    ...notebook.rEnvironment,
    ...parseBackendREnvironment(raw),
  }
}

async function fetchRPackagesAction(): Promise<void> {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  try {
    const data = await strata.getRPackages(sid)
    const packagesRaw = Array.isArray(data?.packages) ? data.packages : []
    const packages = packagesRaw
      .map((p: any) => ({
        name: String(p?.name ?? ''),
        version: String(p?.version ?? ''),
      }))
      .filter((p: { name: string }) => p.name.length > 0)
    const statusRaw = String(data?.packages_status ?? 'unknown')
    const status = (
      ['absent', 'ok', 'rscript_missing', 'renv_not_active', 'failed'].includes(statusRaw)
        ? statusRaw
        : 'unknown'
    ) as RNotebookEnvironment['packagesStatus']
    const errRaw = data?.packages_error
    notebook.rEnvironment = {
      ...notebook.rEnvironment,
      packages,
      packagesStatus: status,
      packagesError: typeof errRaw === 'string' && errRaw.length > 0 ? errRaw : null,
    }
  } catch (err) {
    notebook.rEnvironment = {
      ...notebook.rEnvironment,
      packages: [],
      packagesStatus: 'failed',
      packagesError: (err as Error).message || 'Failed to fetch R packages',
    }
  }
}

function syncEnvironmentPayloadFromBackend(data: any) {
  if (data?.environment) {
    syncNotebookEnvironmentFromBackend(data.environment)
  }
  // R-side env is parallel to Python's. Backend always emits the
  // ``r_environment`` field (with zero fields when the notebook has
  // no renv.lock), so absence here means we're talking to a
  // pre-#87 server — leave the store's default zero entry in
  // place rather than wiping anything.
  if (data?.r_environment) {
    syncNotebookREnvironmentFromBackend(data.r_environment)
  }
  if ('environment_job' in (data || {})) {
    syncEnvironmentOperationFromBackend(data.environment_job)
  }
  if ('environment_job_history' in (data || {})) {
    syncEnvironmentJobHistoryFromBackend(data.environment_job_history)
  }
  if (Array.isArray(data?.dependencies)) {
    dependencies.value = data.dependencies.map((dep: any) => parseDependencyInfo(dep))
  }
  if (Array.isArray(data?.resolved_dependencies)) {
    syncResolvedDependenciesFromBackend(data.resolved_dependencies)
  }
}

function parseWorkerSpec(raw: any): WorkerSpec {
  return {
    name: String(raw?.name || ''),
    backend: raw?.backend === 'executor' ? 'executor' : 'local',
    runtimeId: raw?.runtime_id ?? raw?.runtimeId ?? null,
    config: raw?.config && typeof raw.config === 'object' ? raw.config : {},
  }
}

function parseManagedWorkerSpec(raw: any): ManagedWorkerSpec {
  return {
    ...parseWorkerSpec(raw),
    enabled: raw?.enabled !== false,
  }
}

function parseWorkerTransport(raw: unknown): WorkerCatalogEntry['transport'] | undefined {
  return raw === 'local' ||
    raw === 'embedded' ||
    raw === 'direct' ||
    raw === 'signed' ||
    raw === 'executor'
    ? raw
    : undefined
}

function parseWorkerCatalogEntry(raw: any): WorkerCatalogEntry {
  const transport = parseWorkerTransport(raw?.transport)
  const healthHistory: WorkerHealthHistoryEntry[] = Array.isArray(raw?.health_history)
    ? raw.health_history
        .map((entry: Record<string, unknown>): WorkerHealthHistoryEntry => ({
          checkedAt:
            typeof entry?.checked_at === 'number' && Number.isFinite(entry.checked_at)
              ? entry.checked_at
              : 0,
          health:
            entry?.health === 'healthy' || entry?.health === 'unavailable'
              ? entry.health
              : 'unknown',
          error:
            typeof entry?.error === 'string' && entry.error.trim() ? String(entry.error) : null,
          durationMs:
            typeof entry?.duration_ms === 'number' && Number.isFinite(entry.duration_ms)
              ? entry.duration_ms
              : null,
        }))
        .filter((entry: WorkerHealthHistoryEntry) => entry.checkedAt > 0)
    : []

  return {
    ...parseWorkerSpec(raw),
    health: raw?.health === 'healthy' || raw?.health === 'unavailable' ? raw.health : 'unknown',
    source:
      raw?.source === 'builtin' ||
      raw?.source === 'notebook' ||
      raw?.source === 'server' ||
      raw?.source === 'referenced'
        ? raw.source
        : undefined,
    allowed: raw?.allowed !== false,
    enabled: typeof raw?.enabled === 'boolean' ? raw.enabled : undefined,
    transport,
    healthUrl: typeof raw?.health_url === 'string' ? raw.health_url : null,
    healthCheckedAt:
      typeof raw?.health_checked_at === 'number' && Number.isFinite(raw.health_checked_at)
        ? raw.health_checked_at
        : null,
    lastError: typeof raw?.last_error === 'string' && raw.last_error.trim() ? raw.last_error : null,
    healthHistory,
    probeCount:
      typeof raw?.probe_count === 'number' && Number.isFinite(raw.probe_count)
        ? raw.probe_count
        : 0,
    healthyProbeCount:
      typeof raw?.healthy_probe_count === 'number' && Number.isFinite(raw.healthy_probe_count)
        ? raw.healthy_probe_count
        : 0,
    unavailableProbeCount:
      typeof raw?.unavailable_probe_count === 'number' &&
      Number.isFinite(raw.unavailable_probe_count)
        ? raw.unavailable_probe_count
        : 0,
    unknownProbeCount:
      typeof raw?.unknown_probe_count === 'number' && Number.isFinite(raw.unknown_probe_count)
        ? raw.unknown_probe_count
        : 0,
    consecutiveFailures:
      typeof raw?.consecutive_failures === 'number' && Number.isFinite(raw.consecutive_failures)
        ? raw.consecutive_failures
        : 0,
    lastHealthyAt:
      typeof raw?.last_healthy_at === 'number' && Number.isFinite(raw.last_healthy_at)
        ? raw.last_healthy_at
        : null,
    lastUnavailableAt:
      typeof raw?.last_unavailable_at === 'number' && Number.isFinite(raw.last_unavailable_at)
        ? raw.last_unavailable_at
        : null,
    lastUnknownAt:
      typeof raw?.last_unknown_at === 'number' && Number.isFinite(raw.last_unknown_at)
        ? raw.last_unknown_at
        : null,
    lastStatusChangeAt:
      typeof raw?.last_status_change_at === 'number' && Number.isFinite(raw.last_status_change_at)
        ? raw.last_status_change_at
        : null,
    lastProbeDurationMs:
      typeof raw?.last_probe_duration_ms === 'number' && Number.isFinite(raw.last_probe_duration_ms)
        ? raw.last_probe_duration_ms
        : null,
  }
}

function applyRemoteExecutionMetadata(cell: Cell, raw: Record<string, any>) {
  const remoteWorker =
    typeof raw.remote_worker === 'string' && raw.remote_worker.trim() ? raw.remote_worker : null
  const remoteTransport = parseWorkerTransport(raw.remote_transport) ?? null
  const remoteBuildId =
    typeof raw.remote_build_id === 'string' && raw.remote_build_id.trim()
      ? raw.remote_build_id
      : null
  const remoteBuildState =
    typeof raw.remote_build_state === 'string' && raw.remote_build_state.trim()
      ? raw.remote_build_state
      : null
  const remoteErrorCode =
    typeof raw.remote_error_code === 'string' && raw.remote_error_code.trim()
      ? raw.remote_error_code
      : null

  if (remoteWorker || remoteTransport || remoteBuildId || remoteBuildState || remoteErrorCode) {
    cell.remoteWorkerName = remoteWorker ?? undefined
    cell.remoteTransport = remoteTransport
    cell.remoteBuildId = remoteBuildId
    cell.remoteBuildState = remoteBuildState
    cell.remoteErrorCode = remoteErrorCode
    return
  }

  if (raw.execution_method !== 'cached') {
    cell.remoteWorkerName = undefined
    cell.remoteTransport = null
    cell.remoteBuildId = null
    cell.remoteBuildState = null
    cell.remoteErrorCode = null
  }
}

function applySerializedExecutionMetadata(cell: Cell, raw: Record<string, any>) {
  cell.executorName =
    typeof raw.execution_method === 'string' && raw.execution_method.trim()
      ? raw.execution_method
      : undefined
  cell.remoteWorkerName =
    typeof raw.remote_worker === 'string' && raw.remote_worker.trim()
      ? raw.remote_worker
      : undefined
  cell.remoteTransport = parseWorkerTransport(raw.remote_transport) ?? null
  cell.remoteBuildId =
    typeof raw.remote_build_id === 'string' && raw.remote_build_id.trim()
      ? raw.remote_build_id
      : null
  cell.remoteBuildState =
    typeof raw.remote_build_state === 'string' && raw.remote_build_state.trim()
      ? raw.remote_build_state
      : null
  cell.remoteErrorCode =
    typeof raw.remote_error_code === 'string' && raw.remote_error_code.trim()
      ? raw.remote_error_code
      : null
}

function parseBackendAnnotations(raw: any): CellAnnotations | undefined {
  if (!raw) return undefined
  return {
    name: typeof raw.name === 'string' && raw.name.trim() ? raw.name : null,
    worker: raw.worker ?? null,
    timeout: raw.timeout ?? null,
    env: raw.env || {},
    mounts: Array.isArray(raw.mounts) ? raw.mounts.map(parseMountSpec) : [],
    loop: parseBackendLoopAnnotation(raw.loop),
    variant: parseBackendVariantAnnotation(raw.variant),
  }
}

function parseBackendVariantAnnotation(raw: any): CellAnnotations['variant'] {
  if (!raw || typeof raw !== 'object') return null
  const group = typeof raw.group === 'string' ? raw.group : ''
  const name = typeof raw.name === 'string' ? raw.name : ''
  if (!group || !name) return null
  return { group, name }
}

function parseBackendVariantGroups(raw: any): VariantGroup[] {
  if (!Array.isArray(raw)) return []
  const out: VariantGroup[] = []
  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') continue
    const group = typeof entry.group === 'string' ? entry.group : ''
    const activeName = typeof entry.active_name === 'string' ? entry.active_name : ''
    const activeCellId = typeof entry.active_cell_id === 'string' ? entry.active_cell_id : ''
    if (!group || !activeName || !activeCellId) continue
    const mode = entry.mode === 'sweep' ? 'sweep' : 'switch'
    const members: VariantMember[] = []
    if (Array.isArray(entry.members)) {
      for (const m of entry.members) {
        if (!m || typeof m !== 'object') continue
        const cellId = typeof m.cell_id === 'string' ? m.cell_id : ''
        const name = typeof m.name === 'string' ? m.name : ''
        if (!cellId || !name) continue
        members.push({ cellId, name, isActive: Boolean(m.is_active) })
      }
    }
    out.push({ group, activeName, activeCellId, mode, members })
  }
  return out
}

function parseBackendLoopAnnotation(raw: any): CellAnnotations['loop'] {
  if (!raw || typeof raw !== 'object') return null
  const maxIter = Number(raw.max_iter ?? raw.maxIter ?? 0)
  const carry = typeof raw.carry === 'string' ? raw.carry : ''
  if (!maxIter || !carry) return null
  return {
    maxIter,
    carry,
    untilExpr:
      typeof raw.until_expr === 'string'
        ? raw.until_expr
        : typeof raw.untilExpr === 'string'
          ? raw.untilExpr
          : null,
    startFromCell:
      typeof raw.start_from_cell === 'string'
        ? raw.start_from_cell
        : typeof raw.startFromCell === 'string'
          ? raw.startFromCell
          : null,
    startFromIter:
      raw.start_from_iter !== undefined && raw.start_from_iter !== null
        ? Number(raw.start_from_iter)
        : raw.startFromIter !== undefined && raw.startFromIter !== null
          ? Number(raw.startFromIter)
          : null,
  }
}

function parseEnvMap(raw: any): Record<string, string> {
  if (!raw || typeof raw !== 'object') return {}
  return Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, String(value)]))
}

function parseDisplayOutputPayload(
  raw: any,
  fallbackArtifactUri: string | null = null,
): CellOutput | undefined {
  if (!raw || typeof raw !== 'object') return undefined

  const contentType = String(
    raw.content_type || raw.contentType || 'json/object',
  ) as CellOutput['contentType']
  const output: CellOutput = {
    contentType,
    artifactUri:
      typeof raw.artifact_uri === 'string' && raw.artifact_uri.trim()
        ? raw.artifact_uri
        : fallbackArtifactUri || undefined,
  }

  if (contentType === 'arrow/ipc') {
    const columns = Array.isArray(raw.columns)
      ? raw.columns.map((value: unknown) => String(value))
      : undefined
    const rowCount =
      typeof raw.rows === 'number'
        ? raw.rows
        : typeof raw.row_count === 'number'
          ? raw.row_count
          : typeof raw.rowCount === 'number'
            ? raw.rowCount
            : undefined
    output.columns = columns
    output.rowCount = rowCount

    const preview = raw.preview
    if (Array.isArray(preview) && columns) {
      output.rows = preview.map((row: unknown) => {
        const values = Array.isArray(row) ? row : []
        const obj: Record<string, unknown> = {}
        columns.forEach((col: string, i: number) => {
          obj[col] = values[i]
        })
        return obj
      })
    }
    return output
  }

  if (contentType === 'image/png') {
    output.inlineDataUrl =
      typeof raw.inline_data_url === 'string' && raw.inline_data_url.trim()
        ? raw.inline_data_url
        : null
    output.width = typeof raw.width === 'number' ? raw.width : null
    output.height = typeof raw.height === 'number' ? raw.height : null
    return output
  }

  if (contentType === 'text/markdown') {
    output.markdownText =
      typeof raw.markdown_text === 'string'
        ? raw.markdown_text
        : typeof raw.preview === 'string'
          ? raw.preview
          : null
    return output
  }

  const scalar = raw.data ?? raw.preview
  if (scalar !== undefined) {
    output.scalar = scalar
  }
  return output
}

function parseDisplayOutputPayloads(
  raw: any,
  fallbackArtifactUri: string | null = null,
): CellOutput[] {
  if (!Array.isArray(raw)) return []
  return raw
    .map((entry) => parseDisplayOutputPayload(entry, fallbackArtifactUri))
    .filter((entry): entry is CellOutput => Boolean(entry))
}

function applyDisplayOutputsToCell(
  cell: Cell,
  rawDisplays: any,
  rawDisplay: any,
  fallbackArtifactUri: string | null = null,
) {
  const displayOutputs = parseDisplayOutputPayloads(rawDisplays, fallbackArtifactUri)
  if (!displayOutputs.length) {
    const fallbackOutput = parseDisplayOutputPayload(rawDisplay, fallbackArtifactUri)
    if (fallbackOutput) {
      displayOutputs.push(fallbackOutput)
    }
  }
  cell.displayOutputs = displayOutputs
  cell.output = displayOutputs.at(-1)
}

function applyBackendCellState(localCell: Cell, serverCell: any) {
  localCell.defines = serverCell.defines || []
  localCell.references = serverCell.references || []
  localCell.upstreamIds = serverCell.upstream_ids || []
  localCell.downstreamIds = serverCell.downstream_ids || []
  localCell.isLeaf = serverCell.is_leaf || false
  localCell.status = serverCell.status || 'idle'
  localCell.worker = serverCell.worker ?? null
  localCell.workerOverride = serverCell.worker_override ?? null
  localCell.timeout = serverCell.timeout ?? null
  localCell.timeoutOverride = serverCell.timeout_override ?? null
  localCell.env = parseEnvMap(serverCell.env)
  localCell.envOverrides = parseEnvMap(serverCell.env_overrides)
  localCell.mounts = Array.isArray(serverCell.mounts) ? serverCell.mounts.map(parseMountSpec) : []
  localCell.mountOverrides = Array.isArray(serverCell.mount_overrides)
    ? serverCell.mount_overrides.map(parseMountSpec)
    : []
  localCell.isModuleCell = Boolean(serverCell.is_module_cell)
  localCell.moduleExports = Array.isArray(serverCell.module_exports)
    ? serverCell.module_exports
        .filter((e: any) => e && typeof e.name === 'string')
        .map((e: any) => ({ name: String(e.name), kind: String(e.kind ?? '') }))
    : undefined
  localCell.annotations = parseBackendAnnotations(serverCell.annotations)
  localCell.annotationDiagnostics = Array.isArray(serverCell.annotation_diagnostics)
    ? serverCell.annotation_diagnostics
        .filter((d: any) => d && typeof d.code === 'string')
        .map((d: any) => ({
          severity: d.severity === 'info' ? ('info' as const) : ('warn' as const),
          code: String(d.code),
          message: String(d.message ?? ''),
          line: typeof d.line === 'number' ? d.line : null,
        }))
    : undefined
  localCell.stalenessReasons = supportsStalenessDetail(localCell.status)
    ? serverCell.staleness_reasons || serverCell.staleness?.reasons || []
    : []
  localCell.causality = supportsStalenessDetail(localCell.status)
    ? parseBackendCausality(serverCell.causality)
    : undefined
  localCell.shadowWarnings = Array.isArray(serverCell.shadow_warnings)
    ? serverCell.shadow_warnings
    : undefined
  localCell.widget = parseWidgetSpec(serverCell.widget)
  applyDisplayOutputsToCell(
    localCell,
    serverCell.display_outputs,
    serverCell.display_output,
    typeof serverCell.artifact_uri === 'string' ? serverCell.artifact_uri : null,
  )
  applySerializedExecutionMetadata(localCell, serverCell)
}

function syncCellsFromBackend(serverCells: any[]) {
  for (const serverCell of serverCells) {
    const localCell = cellMap.value.get(serverCell.id)
    if (!localCell) continue
    applyBackendCellState(localCell, serverCell)
  }
}

function syncNotebookMountsFromBackend(serverMounts: any[]) {
  notebook.mounts = Array.isArray(serverMounts) ? serverMounts.map(parseMountSpec) : []
}

function syncNotebookWorkerFromBackend(serverWorker: any) {
  notebook.worker = typeof serverWorker === 'string' && serverWorker.trim() ? serverWorker : null
}

function syncNotebookWorkersFromBackend(serverWorkers: any[]) {
  notebook.workers = Array.isArray(serverWorkers)
    ? serverWorkers.map(parseWorkerSpec).filter((worker) => worker.name)
    : []
}

function syncWorkerCatalogFromBackend(serverWorkers: any[]) {
  availableWorkers.value = Array.isArray(serverWorkers)
    ? serverWorkers.map(parseWorkerCatalogEntry).filter((worker) => worker.name)
    : []
  workerCatalogLoaded.value = true
}

function syncWorkerHealthCheckedAtFromBackend(rawCheckedAt: unknown) {
  if (typeof rawCheckedAt === 'number' && Number.isFinite(rawCheckedAt)) {
    workerHealthCheckedAt.value = rawCheckedAt
  }
}

function updateWorkerHealth(workerName: string, health: WorkerHealth) {
  availableWorkers.value = applyWorkerHealth(availableWorkers.value, workerName, health)
}

function syncWorkerDefinitionsEditableFromBackend(value: any) {
  // Fail closed: only treat as editable (personal mode) when the backend
  // explicitly says so. Unknown / missing / errored responses default to
  // service mode so the UI doesn't misreport mode before the first sync.
  workerDefinitionsEditable.value = value === true
}

function syncServerManagedWorkersFromBackend(serverWorkers: any[]) {
  serverManagedWorkers.value = Array.isArray(serverWorkers)
    ? serverWorkers.map(parseManagedWorkerSpec).filter((worker) => worker.name)
    : []
}

function clearServerWorkerRegistryState() {
  serverManagedWorkers.value = []
  serverWorkerRegistryAvailable.value = false
  serverWorkerRegistryLoading.value = false
  serverWorkerActionLoading.value = {}
  serverWorkerRegistryError.value = null
}

function resetWorkerCatalogState() {
  availableWorkers.value = []
  workerCatalogLoaded.value = false
  // Fail closed to service mode; backend will re-sync personal if applicable.
  workerDefinitionsEditable.value = false
  workerHealthLoading.value = false
  workerHealthCheckedAt.value = null
  notebookWorkerError.value = null
  workerRegistryError.value = null
  clearServerWorkerRegistryState()
}

function syncNotebookTimeoutFromBackend(serverTimeout: any) {
  notebook.timeout = typeof serverTimeout === 'number' ? serverTimeout : null
}

function syncNotebookEnvFromBackend(serverEnv: any) {
  notebook.env = parseEnvMap(serverEnv)
}

/** Merge the env-response fields (sources + fetch status + secret-manager
 * config) into the store. Shared by the env PUT, /secret-manager/refresh,
 * /secret-manager/config PUT, and initial open since all four share shape. */
function applyEnvSources(payload: any) {
  notebook.envSources =
    payload && typeof payload.env_sources === 'object'
      ? Object.fromEntries(
          Object.entries(payload.env_sources as Record<string, unknown>)
            .filter(([, v]) => typeof v === 'string')
            .map(([k, v]) => [k, String(v)]),
        )
      : {}
  notebook.envFetchError =
    typeof payload?.env_fetch_error === 'string' ? payload.env_fetch_error : null
  notebook.envFetchedAt =
    typeof payload?.env_fetched_at === 'string' ? payload.env_fetched_at : null
  notebook.secretManagerConfig =
    payload && typeof payload.secret_manager_config === 'object' && payload.secret_manager_config
      ? Object.fromEntries(
          Object.entries(payload.secret_manager_config as Record<string, unknown>)
            .filter(([, v]) => typeof v === 'string')
            .map(([k, v]) => [k, String(v)]),
        )
      : {}
}

function syncNotebookEnvironmentFromBackend(serverEnvironment: any) {
  notebook.environment = parseBackendEnvironment(serverEnvironment)
}

function parseBackendNotebookRuntimeConfig(raw: any): NotebookRuntimeConfig {
  const availablePythonVersions = Array.isArray(raw?.available_python_versions)
    ? raw.available_python_versions
        .map((value: unknown) => String(value || '').trim())
        .filter((value: string) => value.length > 0)
    : []
  const defaultPythonVersion =
    typeof raw?.default_python_version === 'string' && raw.default_python_version.trim()
      ? raw.default_python_version
      : availablePythonVersions[0] || ''
  return {
    // Fail closed to 'service' so the header doesn't falsely advertise
    // personal mode before the runtime-defaults fetch succeeds. Personal
    // is only reported when the backend explicitly says so.
    deploymentMode: raw?.deployment_mode === 'personal' ? 'personal' : 'service',
    defaultParentPath:
      typeof raw?.default_parent_path === 'string' && raw.default_parent_path.trim()
        ? raw.default_parent_path
        : FALLBACK_NOTEBOOK_PARENT_PATH,
    availablePythonVersions,
    defaultPythonVersion,
    pythonSelectionFixed:
      raw?.python_selection_fixed === true || availablePythonVersions.length <= 1,
    // Fail closed: registry UI only shows when the backend explicitly enables it.
    registryEnabled: raw?.registry_enabled === true,
  }
}

function setEnvironmentActionSummary(raw: {
  action: EnvironmentJobAction
  packageName?: string | null
  lockfileChanged?: boolean
  staleCellCount?: number
}) {
  environmentLastAction.value = {
    action: raw.action,
    packageName: raw.packageName ?? null,
    lockfileChanged: raw.lockfileChanged === true,
    staleCellCount: raw.staleCellCount ?? 0,
    timestamp: Date.now(),
  }
}

function extractOperationLogPayload(raw: any): any | null {
  if (raw && typeof raw === 'object') {
    if (raw.operation_log && typeof raw.operation_log === 'object') {
      return raw.operation_log
    }
    if (raw.detail && typeof raw.detail === 'object') {
      const nested = (raw.detail as Record<string, unknown>).operation_log
      if (nested && typeof nested === 'object') {
        return nested
      }
    }
  }
  return null
}

function beginEnvironmentOperation(
  action: EnvironmentOperation['action'],
  command: string,
  packageName: string | null = null,
) {
  environmentOperation.value = {
    id: `${action}-${Date.now()}`,
    action,
    packageName,
    status: 'running',
    phase: 'starting',
    command,
    durationMs: null,
    stdout: '',
    stderr: '',
    stdoutTruncated: false,
    stderrTruncated: false,
    startedAt: Date.now(),
    finishedAt: null,
    lockfileChanged: false,
    staleCellCount: 0,
    staleCellIds: [],
    error: null,
  }
}

function finishEnvironmentOperation(
  action: EnvironmentOperation['action'],
  status: EnvironmentOperation['status'],
  raw: any,
  fallbackCommand: string,
) {
  const previous = environmentOperation.value
  environmentOperation.value = {
    id: previous?.id || `${action}-${Date.now()}`,
    action,
    packageName: previous?.packageName ?? null,
    status,
    phase: status === 'running' ? previous?.phase || 'running' : status,
    command: typeof raw?.command === 'string' && raw.command.trim() ? raw.command : fallbackCommand,
    durationMs:
      typeof raw?.duration_ms === 'number'
        ? raw.duration_ms
        : action === 'sync'
          ? notebook.environment.lastSyncDurationMs
          : null,
    stdout: typeof raw?.stdout === 'string' ? raw.stdout : '',
    stderr: typeof raw?.stderr === 'string' ? raw.stderr : '',
    stdoutTruncated: raw?.stdout_truncated === true,
    stderrTruncated: raw?.stderr_truncated === true,
    startedAt: previous?.action === action ? previous.startedAt : Date.now(),
    finishedAt: status === 'running' ? null : Date.now(),
    lockfileChanged: previous?.lockfileChanged === true,
    staleCellCount: previous?.staleCellCount ?? 0,
    staleCellIds: previous?.staleCellIds ?? [],
    error: status === 'failed' ? (typeof raw?.error === 'string' ? raw.error : null) : null,
  }
}

function parseWidgetSpec(raw: any): WidgetSpec | undefined {
  if (!raw || !Array.isArray(raw.descriptors)) return undefined
  return {
    descriptors: raw.descriptors.map((d: any) => ({
      name: String(d.name),
      kind: d.kind,
      params: d.params && typeof d.params === 'object' ? d.params : {},
      default: d.default,
    })),
    values: raw.values && typeof raw.values === 'object' ? raw.values : {},
  }
}

function parseBackendCellPayload(raw: any): Cell {
  const cell: Cell = {
    id: raw.id,
    source: raw.source || '',
    language: raw.language || 'python',
    order: raw.order ?? 0,
    status: (raw.status || 'idle') as CellStatus,
    worker: raw.worker ?? null,
    workerOverride: raw.worker_override ?? null,
    timeout: raw.timeout ?? null,
    timeoutOverride: raw.timeout_override ?? null,
    env: parseEnvMap(raw.env),
    envOverrides: parseEnvMap(raw.env_overrides),
    mounts: Array.isArray(raw.mounts) ? raw.mounts.map(parseMountSpec) : [],
    mountOverrides: Array.isArray(raw.mount_overrides)
      ? raw.mount_overrides.map(parseMountSpec)
      : [],
    annotations: parseBackendAnnotations(raw.annotations),
    upstreamIds: raw.upstream_ids || [],
    downstreamIds: raw.downstream_ids || [],
    defines: raw.defines || [],
    references: raw.references || [],
    inputs: [],
    isLeaf: raw.is_leaf || false,
    stalenessReasons: supportsStalenessDetail(raw.status)
      ? raw.staleness_reasons || raw.staleness?.reasons || []
      : [],
    causality: supportsStalenessDetail(raw.status)
      ? parseBackendCausality(raw.causality)
      : undefined,
    output: undefined,
    displayOutputs: [],
    variantGroup: typeof raw.variant_group === 'string' ? raw.variant_group : null,
    variantName: typeof raw.variant_name === 'string' ? raw.variant_name : null,
    variantActive: raw.variant_active !== false,
    widget: parseWidgetSpec(raw.widget),
  }

  applyDisplayOutputsToCell(
    cell,
    raw.display_outputs,
    raw.display_output,
    typeof raw.artifact_uri === 'string' ? raw.artifact_uri : null,
  )

  applySerializedExecutionMetadata(cell, raw)

  // Restore persisted console output so it survives notebook reopens.
  // Stored on the Cell directly — the renderer shows it in its own
  // panel below the cell's structured output so it doesn't pollute
  // the display value (important for @output_schema cells where the
  // output is a user-controlled JSON object).
  cell.consoleStdout = typeof raw.console_stdout === 'string' ? raw.console_stdout : ''
  cell.consoleStderr = typeof raw.console_stderr === 'string' ? raw.console_stderr : ''

  // Unit-test source + last result (Python cells). Hydrated on open so the
  // panel and the toolbar health badge restore without a re-run.
  cell.testSource = typeof raw.test_source === 'string' ? raw.test_source : ''
  cell.testResult = raw.test_result ? parseBackendTestResult(raw.test_result) : undefined

  return cell
}

function loadNotebookStateFromBackend(data: any) {
  notebook.id = data.id
  notebook.name = data.name
  notebook.worker = data.worker ?? null
  notebook.timeout = data.timeout ?? null
  notebook.env = parseEnvMap(data.env)
  // Registry UI gate from the runtime config merged into the open response.
  registryEnabled.value = data?.registry_enabled === true
  // Populate the per-cell strips for already-published artifacts on open.
  scheduleRegistryRefresh()
  applyEnvSources(data)
  notebook.workers = Array.isArray(data.workers) ? data.workers.map(parseWorkerSpec) : []
  notebook.mounts = Array.isArray(data.mounts) ? data.mounts.map(parseMountSpec) : []
  notebook.connections = Array.isArray(data.connections)
    ? (data.connections as ConnectionSpec[])
    : []
  syncEnvironmentPayloadFromBackend(data)
  notebook.createdAt = data.created_at ? new Date(data.created_at).getTime() : Date.now()
  notebook.updatedAt = data.updated_at ? new Date(data.updated_at).getTime() : Date.now()
  ;(notebook as any).sessionId = data.session_id

  notebook.cells = (data.cells || []).map(parseBackendCellPayload)
  notebook.cells.sort((a, b) => a.order - b.order)
  notebook.variantGroups = parseBackendVariantGroups(data.variant_groups)
  if (data.dag) {
    applyBackendDag(data.dag)
  }
}

async function moveCell(id: CellId, direction: 'up' | 'down') {
  const sorted = orderedCells.value
  const idx = sorted.findIndex((c) => c.id === id)
  const swapIdx = direction === 'up' ? idx - 1 : idx + 1
  if (swapIdx < 0 || swapIdx >= sorted.length) return

  // Swap locally for instant feedback
  const tmp = sorted[idx].order
  sorted[idx].order = sorted[swapIdx].order
  sorted[swapIdx].order = tmp
  notebook.updatedAt = Date.now()

  // Persist to backend
  const sid = sessionId()
  if (!sid) return
  try {
    const newOrder = orderedCells.value.map((c) => c.id)
    await useStrata().reorderCells(sid, newOrder)
  } catch (err) {
    // Revert on failure
    const revertTmp = sorted[idx].order
    sorted[idx].order = sorted[swapIdx].order
    sorted[swapIdx].order = revertTmp
    console.error('Failed to reorder cells:', err)
  }
}

async function duplicateCell(id: CellId) {
  const cell = cellMap.value.get(id)
  if (!cell) return
  const source = cell.source
  await addCell(id)
  // Find the newly added cell (right after the original)
  const sorted = orderedCells.value
  const idx = sorted.findIndex((c) => c.id === id)
  const newCell = sorted[idx + 1]
  if (newCell && source) {
    await updateSource(newCell.id, source)
  }
}

function applyBackendDag(backendDag: any) {
  for (const cell of notebook.cells) {
    cell.upstreamIds = []
    cell.downstreamIds = []
    cell.isLeaf = backendDag.leaves?.includes(cell.id) || false
  }
  const edges: DagEdge[] = []
  if (backendDag.edges && Array.isArray(backendDag.edges)) {
    for (const edge of backendDag.edges) {
      const fromCell = cellMap.value.get(edge.from_cell_id)
      const toCell = cellMap.value.get(edge.to_cell_id)
      if (fromCell && toCell) {
        if (!toCell.upstreamIds.includes(edge.from_cell_id)) {
          toCell.upstreamIds.push(edge.from_cell_id)
        }
        if (!fromCell.downstreamIds.includes(edge.to_cell_id)) {
          fromCell.downstreamIds.push(edge.to_cell_id)
        }
        edges.push({
          from_cell_id: edge.from_cell_id,
          to_cell_id: edge.to_cell_id,
          variable: typeof edge.variable === 'string' ? edge.variable : '',
        })
      }
    }
  }
  backendDagEdges.value = edges
  if (backendDag.variant_groups !== undefined) {
    notebook.variantGroups = parseBackendVariantGroups(backendDag.variant_groups)
    syncCellVariantFlagsFromGroups()
  }
}

/** Re-derive cell.variantActive from notebook.variantGroups so cells and
 *  groups never drift. The backend is authoritative; this just mirrors. */
function syncCellVariantFlagsFromGroups() {
  const activeIds = new Set<CellId>()
  for (const group of notebook.variantGroups) {
    activeIds.add(group.activeCellId)
  }
  const groupedIds = new Set<CellId>()
  for (const group of notebook.variantGroups) {
    for (const member of group.members) {
      groupedIds.add(member.cellId)
    }
  }
  for (const cell of notebook.cells) {
    if (!groupedIds.has(cell.id)) continue
    cell.variantActive = activeIds.has(cell.id)
  }
}

// --- API Integration -------------------------------------------------------

/**
 * Boot: create scratch notebook with one empty cell, connect WebSocket.
 * Called once on app mount. Resolves when ready to use.
 */
async function boot(): Promise<void> {
  const strata = useStrata()
  try {
    markNotebookPerf('boot_request_start')
    connectError.value = null
    dependencyError.value = null
    environmentError.value = null
    environmentWarnings.value = []
    environmentLastAction.value = null
    environmentOperation.value = null
    environmentJobHistory.value = []
    resetWorkerCatalogState()
    const runtimeConfig = parseBackendNotebookRuntimeConfig(await strata.getNotebookRuntimeConfig())

    // Create a scratch notebook with one starter cell
    const data = await strata.createNotebook(runtimeConfig.defaultParentPath, 'scratch', null, true)
    markNotebookPerf('boot_response')
    measureNotebookPerf('boot_request_ms', 'boot_request_start', 'boot_response')
    loadNotebookStateFromBackend(data)
    markNotebookPerf('boot_hydrated')
    measureNotebookPerf('boot_hydrate_ms', 'boot_response', 'boot_hydrated')

    // Connect WebSocket in the background; initial notebook render does not
    // need to wait for the live channel as long as execution stays disabled.
    initializeWebSocket()
    connected.value = false
    void waitForWebSocket()
      .then(() => {
        markNotebookPerf('boot_ws_ready')
        measureNotebookPerf('boot_ws_connect_ms', 'boot_hydrated', 'boot_ws_ready')
        measureNotebookPerf('boot_total_ms', 'boot_request_start', 'boot_ws_ready')
        connectError.value = null
        connected.value = true
      })
      .catch((e: any) => {
        connectError.value = e?.message || 'Failed to connect to server'
        connected.value = false
      })
  } catch (e: any) {
    console.error('Failed to boot notebook:', e)
    connectError.value = e.message || 'Failed to connect to server'
    connected.value = false
  }
}

/**
 * Open an existing notebook from disk.
 */
async function openNotebook(path: string): Promise<any> {
  const strata = useStrata()
  markNotebookPerf('store_open_request_start')
  // Cleanup existing WebSocket
  cleanupWebSocket()
  dependencyError.value = null
  environmentError.value = null
  environmentWarnings.value = []
  environmentLastAction.value = null
  environmentOperation.value = null
  environmentJobHistory.value = []
  resetWorkerCatalogState()

  const data = await strata.openNotebook(path)
  markNotebookPerf('store_open_response')
  measureNotebookPerf('store_open_request_ms', 'store_open_request_start', 'store_open_response')
  loadNotebookStateFromBackend(data)
  markNotebookPerf('store_open_hydrated')
  measureNotebookPerf('store_open_hydrate_ms', 'store_open_response', 'store_open_hydrated')

  initializeWebSocket()
  connected.value = false
  void waitForWebSocket()
    .then(() => {
      markNotebookPerf('store_open_ws_ready')
      measureNotebookPerf('store_open_ws_connect_ms', 'store_open_hydrated', 'store_open_ws_ready')
      measureNotebookPerf('store_open_total_ms', 'store_open_request_start', 'store_open_ws_ready')
      connectError.value = null
      connected.value = true
    })
    .catch((e: any) => {
      connectError.value = e?.message || 'Failed to connect to server'
      connected.value = false
    })

  return data
}

/**
 * Reconnect to an existing session by session ID.
 * Used when navigating to /notebook/:sessionId (e.g. page refresh).
 * Returns session data including path/name for recent-notebooks tracking.
 */
async function openBySessionId(sessionId: string): Promise<any> {
  const strata = useStrata()
  markNotebookPerf('store_session_request_start')
  cleanupWebSocket()
  dependencyError.value = null
  environmentError.value = null
  environmentWarnings.value = []
  environmentLastAction.value = null
  environmentOperation.value = null
  environmentJobHistory.value = []
  resetWorkerCatalogState()

  const data = consumePrefetchedNotebookSession(sessionId) ?? (await strata.getSession(sessionId))
  markNotebookPerf('store_session_response')
  measureNotebookPerf(
    'store_session_request_ms',
    'store_session_request_start',
    'store_session_response',
  )
  if (!data.session_id) {
    data.session_id = sessionId
  }
  loadNotebookStateFromBackend(data)
  markNotebookPerf('store_session_hydrated')
  measureNotebookPerf(
    'store_session_hydrate_ms',
    'store_session_response',
    'store_session_hydrated',
  )

  initializeWebSocket()
  connected.value = false
  void waitForWebSocket()
    .then(() => {
      markNotebookPerf('store_session_ws_ready')
      measureNotebookPerf(
        'store_session_ws_connect_ms',
        'store_session_hydrated',
        'store_session_ws_ready',
      )
      measureNotebookPerf(
        'store_session_total_ms',
        'store_session_request_start',
        'store_session_ws_ready',
      )
      connectError.value = null
      connected.value = true
    })
    .catch((e: any) => {
      connectError.value = e?.message || 'Failed to connect to server'
      connected.value = false
    })

  return data
}

async function deleteNotebookAction(): Promise<any> {
  const sid = sessionId()
  if (!sid) {
    throw new Error('Notebook is not open')
  }

  const strata = useStrata()
  const data = await strata.deleteNotebook(sid)

  cleanupWebSocket()
  connected.value = false
  connectError.value = null
  dependencyError.value = null
  environmentError.value = null
  environmentWarnings.value = []
  environmentLastAction.value = null
  environmentOperation.value = null
  environmentJobHistory.value = []
  resetWorkerCatalogState()
  ;(notebook as any).sessionId = undefined

  return data
}

async function updateNotebookNameAction(name: string): Promise<any> {
  const sid = sessionId()
  if (!sid) {
    throw new Error('Notebook is not open')
  }

  const strata = useStrata()
  const data = await strata.renameNotebook(sid, name)
  if (typeof data?.name === 'string' && data.name.trim()) {
    notebook.name = data.name
  }
  return data
}

// --- WebSocket integration -------------------------------------------------

// v1.1: Impact preview and profiling state
const currentImpactPreview = ref<ImpactPreview | null>(null)
const profilingSummary = ref<ProfilingSummary | null>(null)

// Environment / dependency state
const dependencies = ref<DependencyInfo[]>([])
const resolvedDependencies = ref<DependencyInfo[]>([])
const dependencyLoading = ref(false)
const dependencyError = ref<string | null>(null)
const environmentLoading = ref(false)
const environmentError = ref<string | null>(null)
const environmentWarnings = ref<string[]>([])
const environmentLastAction = ref<EnvironmentActionSummary | null>(null)
const environmentOperation = ref<EnvironmentOperation | null>(null)
const environmentJobHistory = ref<EnvironmentOperation[]>([])
const environmentImportPreview = ref<EnvironmentImportPreview | null>(null)
const environmentMutationActive = computed(() => environmentOperation.value?.status === 'running')
const availableWorkers = ref<WorkerCatalogEntry[]>([])
const workerCatalogLoaded = ref(false)
// Default to service mode (non-editable) until the backend confirms
// otherwise. Personal mode relaxes restrictions, so failing closed here
// avoids the header briefly advertising the wrong mode on load.
const workerDefinitionsEditable = ref(false)
// Registry UI gate: true only when the backend reports the registry routes
// are reachable (personal mode today). Fail closed so the dashboard stays
// hidden until the open response confirms it.
const registryEnabled = ref(false)
const workerHealthLoading = ref(false)
const workerHealthCheckedAt = ref<number | null>(null)

// Registry / dashboard state (P3). Populated lazily by the Registry tab and
// the per-cell strip; reads/writes go through the SERVER registry routes.
const registryArtifactsByCell = ref<Record<string, PublishedArtifact[]>>({})
const registryNames = ref<RegistryName[]>([])
const registryPending = ref<PendingChange[]>([])
const registryAudit = ref<AuditEntry[]>([])
const registryLoading = ref(false)
const registryError = ref<string | null>(null)

// Minimal toast feed for registry feedback (promoted / pending / approved).
interface Toast {
  id: number
  kind: 'success' | 'error' | 'info'
  message: string
}
const toasts = ref<Toast[]>([])
let _toastSeq = 0
function pushToast(message: string, kind: Toast['kind'] = 'success', durationMs = 3500) {
  const id = ++_toastSeq
  toasts.value.push({ id, kind, message })
  if (durationMs > 0) {
    setTimeout(() => {
      toasts.value = toasts.value.filter((t) => t.id !== id)
    }, durationMs)
  }
}
function dismissToast(id: number) {
  toasts.value = toasts.value.filter((t) => t.id !== id)
}

// Refresh the registry state shown in the tab + per-cell strip: a cell's
// published artifacts and the pending-approval queue. Called on tab open,
// after a mutation, and after a cell finishes (it may have published).
async function refreshRegistryAction() {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  registryLoading.value = true
  registryError.value = null
  try {
    const [arts, summary, pending] = await Promise.all([
      strata.getNotebookArtifacts(sid),
      strata.getRegistrySummary(),
      strata.getPendingChanges(),
    ])
    registryArtifactsByCell.value = arts.cells || {}
    registryNames.value = summary.names || []
    registryPending.value = pending.pending || []
  } catch (err) {
    registryError.value = err instanceof Error ? err.message : 'Failed to load registry'
  } finally {
    registryLoading.value = false
  }
}

// Debounced refresh: the per-cell strip and tab need fresh data after a cell
// finishes (it may have published) without per-output thrash. No-op when the
// registry is unreachable (service mode).
let _registryRefreshTimer: ReturnType<typeof setTimeout> | null = null
function scheduleRegistryRefresh() {
  if (!registryEnabled.value) return
  if (_registryRefreshTimer) clearTimeout(_registryRefreshTimer)
  _registryRefreshTimer = setTimeout(() => {
    _registryRefreshTimer = null
    void refreshRegistryAction()
  }, 600)
}

async function fetchRegistryAuditAction(name?: string) {
  try {
    registryAudit.value = (await useStrata().getRegistryAudit(name)).entries || []
  } catch (err) {
    registryError.value = err instanceof Error ? err.message : 'Failed to load audit'
  }
}

// Mutations propagate errors so the component can toast; they refresh on
// success. setAlias returns the move result (applied | pending | unchanged).
async function setAliasAction(
  name: string,
  alias: string,
  artifactId: string,
  version: number,
): Promise<AliasMoveResult> {
  const result = await useStrata().setAlias(name, alias, artifactId, version)
  await refreshRegistryAction()
  return result
}

async function approvePendingAction(name: string, alias: string) {
  await useStrata().approvePending(name, alias)
  await refreshRegistryAction()
}

async function rejectPendingAction(name: string, alias: string) {
  await useStrata().rejectPending(name, alias)
  await refreshRegistryAction()
}

function fetchLineageAction(artifactId: string, version: number) {
  return useStrata().getLineage(artifactId, version)
}

// Lineage modal (P3e): fetch the flat graph, adapt to a tree, flatten to
// indented rows, and open the overlay.
const lineageOpen = ref(false)
const lineageTitle = ref('')
const lineageRows = ref<Array<LineageTreeNode & { depth: number }>>([])
const lineageLoading = ref(false)
const lineageError = ref<string | null>(null)

async function openLineageAction(artifactId: string, version: number, title: string) {
  lineageOpen.value = true
  lineageTitle.value = title
  lineageRows.value = []
  lineageError.value = null
  lineageLoading.value = true
  try {
    const graph = await useStrata().getLineage(artifactId, version)
    lineageRows.value = flattenLineage(lineageToTree(graph, graph.artifact_uri))
  } catch (err) {
    lineageError.value = err instanceof Error ? err.message : 'Failed to load lineage'
  } finally {
    lineageLoading.value = false
  }
}

function closeLineage() {
  lineageOpen.value = false
}
const notebookWorkerError = ref<string | null>(null)
const workerRegistryError = ref<string | null>(null)
const serverManagedWorkers = ref<ManagedWorkerSpec[]>([])
const serverWorkerRegistryAvailable = ref(false)
const serverWorkerRegistryLoading = ref(false)
const serverWorkerActionLoading = ref<Record<string, boolean>>({})
const serverWorkerRegistryError = ref<string | null>(null)

// Inspect REPL state
interface InspectEntry {
  expr: string
  result?: string
  error?: string
  type?: string
  stdout?: string
}
// Inspect REPL state is per-cell. The backend supports N concurrent
// inspect sessions per notebook; the frontend caps the visible panels
// via maxInspectPanels (user-configurable, default 1). When the cap is
// hit, opening a new inspect prompts the user to confirm closing the
// oldest panel — see pendingInspectRequest.
const inspectOpenOrder = ref<CellId[]>([])
const inspectReadyMap = ref(new Map<CellId, boolean>())
const inspectHistoryMap = ref(new Map<CellId, InspectEntry[]>())
// In-flight open requests that haven't received an 'inspect_result' yet.
// Counted against the panel cap to prevent a race where two quick clicks
// both pass the check before either response arrives, letting both panels
// through and silently exceeding the cap.
const inspectPendingOpens = ref(new Set<CellId>())
const pendingInspectRequest = ref<{ newCellId: CellId; evictCellId: CellId } | null>(null)

function inspectActiveCount(): number {
  return inspectOpenOrder.value.length + inspectPendingOpens.value.size
}

function readInspectLimit(): number {
  const raw = Number.parseInt(localStorage.getItem('strata.inspect.maxPanels') || '1', 10)
  if (!Number.isFinite(raw) || raw < 1) return 1
  return Math.min(raw, 8)
}
const maxInspectPanels = ref(readInspectLimit())
function setMaxInspectPanels(n: number) {
  const clamped = Math.max(1, Math.min(8, Math.floor(n)))
  maxInspectPanels.value = clamped
  localStorage.setItem('strata.inspect.maxPanels', String(clamped))
}

// Back-compat aliases for callers that still read a single inspect.
// The first open cell (if any) is treated as "the current inspect" for
// legacy code paths. Remove once no consumers reference these.
const inspectCellId = computed<CellId | null>(() => inspectOpenOrder.value[0] ?? null)
const inspectReady = computed(() => {
  const id = inspectCellId.value
  return id ? inspectReadyMap.value.get(id) === true : false
})
const inspectHistory = computed<InspectEntry[]>(() => {
  const id = inspectCellId.value
  return id ? (inspectHistoryMap.value.get(id) ?? []) : []
})

function isInspecting(cellId: CellId): boolean {
  return inspectOpenOrder.value.includes(cellId)
}
function inspectReadyFor(cellId: CellId): boolean {
  return inspectReadyMap.value.get(cellId) === true
}
function inspectHistoryFor(cellId: CellId): InspectEntry[] {
  return inspectHistoryMap.value.get(cellId) ?? []
}

// --- Cell unit-test panels -------------------------------------------------
// Which cells currently have the Tests panel open. Purely client-side
// (mirrors the inspect toggle's UX, but there's no backend open/close — the
// only backend round-trip is running the tests). Test source / result /
// status live on the Cell itself, like console output.
const testOpenCells = ref<Set<CellId>>(new Set())

function isTesting(cellId: CellId): boolean {
  return testOpenCells.value.has(cellId)
}
function toggleTests(cellId: CellId) {
  const next = new Set(testOpenCells.value)
  if (next.has(cellId)) next.delete(cellId)
  else next.add(cellId)
  testOpenCells.value = next
}
function closeTests(cellId: CellId) {
  if (!testOpenCells.value.has(cellId)) return
  const next = new Set(testOpenCells.value)
  next.delete(cellId)
  testOpenCells.value = next
}
function updateTestSource(cellId: CellId, source: string) {
  const cell = cellMap.value.get(cellId)
  if (!cell) return
  cell.testSource = source
  // A test-source edit invalidates the last result.
  if (cell.testResult && !cell.testResult.stale) {
    cell.testResult = { ...cell.testResult, stale: true }
  }
}
function runCellTests(cellId: CellId) {
  const cell = cellMap.value.get(cellId)
  if (!cell || !wsInstance || !wsInstance.connected()) return
  cell.testStatus = 'running'
  wsInstance.runCellTests(cellId, cell.testSource ?? '')
}

function parseBackendTestResult(raw: any): CellTestResult {
  return {
    passed: raw.passed ?? 0,
    failed: raw.failed ?? 0,
    errored: raw.errored ?? 0,
    skipped: raw.skipped ?? 0,
    tests: Array.isArray(raw.tests)
      ? raw.tests.map((t: any) => ({
          name: t.name ?? '',
          nodeid: t.nodeid ?? '',
          outcome: t.outcome ?? 'error',
          message: t.message ?? '',
        }))
      : [],
    // Persisted results carry no `stale` (it's computed at emit time); a
    // freshly hydrated result is trusted until the user edits the cell/tests.
    stale: raw.stale === true,
    pytestUnavailable: raw.pytest_unavailable === true,
    ranAt: raw.ran_at ?? 0,
  }
}

// LLM assistant state
interface LlmMessage {
  role: 'user' | 'assistant'
  content: string
  model?: string
  tokens?: { input: number; output: number }
  timestamp: number
  streaming?: boolean
  // Tracks which entry point produced the message. Chat is
  // conversation-only; agent messages may surface insertable code blocks
  // alongside the agent's own tool-driven cell edits.
  source?: 'chat' | 'agent'
}
const llmAvailable = ref(false)
const llmModel = ref<string | null>(null)
const llmProvider = ref<string | null>(null)
const llmLoading = ref(false)
const llmError = ref<string | null>(null)
const llmMessages = ref<LlmMessage[]>([])

// How many prior turns to send back to the LLM. System prompt + notebook
// context is already heavy; keep history cap modest.
const LLM_HISTORY_LIMIT = 10

// Agent state
interface AgentProgressEvent {
  event: string
  detail: string
  timestamp: number
}

// Pending approval prompts emitted by the backend when the agent wants
// to run a destructive tool (delete_cell, add_package). The user must
// resolve each one before the agent loop continues.
interface AgentApprovalPrompt {
  request_id: string
  tool: string
  arguments: Record<string, any>
  summary: string
}

const agentRunning = ref(false)
const agentProgress = ref<AgentProgressEvent[]>([])
const agentError = ref<string | null>(null)
const agentApprovalPrompts = ref<AgentApprovalPrompt[]>([])
const agentAutoApprove = ref(false)

let wsInstance: ReturnType<typeof useWebSocket> | null = null

// App-view (read-only) mode: when true, the WS connects with `?role=viewer`
// so the server rejects any mutation frame. Set before opening the session.
const viewerMode = ref(false)

function setViewerMode(on: boolean) {
  viewerMode.value = on
}

function initializeWebSocket() {
  if (!wsInstance) {
    const notebookId = (notebook as any).sessionId
    if (!notebookId) return

    wsInstance = useWebSocket(notebookId, { role: viewerMode.value ? 'viewer' : undefined })

    wsInstance.onMessage('cell_status', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const cellId = p.cell_id as CellId
      const status = p.status as CellStatus
      setCellStatus(cellId, status)

      const cell = cellMap.value.get(cellId)

      // A fresh "running" from the backend starts a new execution —
      // drop any stale stream buffer from a previous (e.g. cancelled)
      // run. This covers run-all / rerun-all / cascade / force paths
      // that don't go through the local execute* helpers, where a new
      // attempt-1 stream would otherwise append onto the old text.
      if (cell && status === 'running') {
        cell.streamBuffer = undefined
        cell.streamAttempt = undefined
      }

      if (cell && status !== 'error' && cell.suggestInstall) {
        cell.suggestInstall = undefined
        if (cell.output?.error) {
          cell.displayOutputs = []
          cell.output = undefined
        }
      }

      // When a cell enters "running" state against a remote worker, the
      // backend includes remote_worker + remote_transport on the payload
      // so the UI can render a live "dispatching → X" badge while the
      // cell executes. Local cells omit these fields.
      if (cell && status === 'running' && typeof p.remote_worker === 'string') {
        applyRemoteExecutionMetadata(cell, p)
      }

      if (cell && p.causality && supportsStalenessDetail(status)) {
        const rawCausality = p.causality as Record<string, any>
        cell.causality = {
          reason: rawCausality.reason,
          details: (rawCausality.details || []).map((d: Record<string, any>) => ({
            type: d.type,
            cellId: d.cell_id,
            cellName: d.cell_name,
            fromVersion: d.from_version,
            toVersion: d.to_version,
            package: d.package,
            fromPackageVersion: d.from_package_version,
            toPackageVersion: d.to_package_version,
          })),
        }
      } else if (cell) {
        cell.causality = undefined
      }

      if (cell && supportsStalenessDetail(status) && p.staleness_reasons) {
        cell.stalenessReasons = p.staleness_reasons
      } else if (cell && !supportsStalenessDetail(status)) {
        cell.stalenessReasons = []
      }
    })

    wsInstance.onMessage('cell_output', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const cellId = p.cell_id as CellId
      const outputs = p.outputs as Record<string, any> | undefined
      // A finished cell may have published to the registry — refresh the
      // per-cell strip + Registry tab (debounced, registry-gated).
      scheduleRegistryRefresh()

      const displayOutputs =
        parseDisplayOutputPayloads(
          p.displays,
          typeof p.artifact_uri === 'string' ? p.artifact_uri : null,
        ) || []
      if (!displayOutputs.length) {
        const displayPayload =
          p.display ?? (outputs && typeof outputs === 'object' ? outputs['_'] : null)
        const fallbackOutput = parseDisplayOutputPayload(
          displayPayload,
          typeof p.artifact_uri === 'string' ? p.artifact_uri : null,
        )
        if (fallbackOutput) {
          displayOutputs.push(fallbackOutput)
        }
      }

      let output: CellOutput = displayOutputs.at(-1) ?? {
        contentType: 'json/object',
        artifactUri: p.artifact_uri,
      }
      output.cacheHit = p.cache_hit || false
      output.cacheLoadMs = p.duration_ms

      // Console output is stored on the Cell, not on output.scalar —
      // keeps the structured display value clean for @output_schema
      // prompts and pandas cells that return typed objects.
      const stdout = typeof p.stdout === 'string' ? p.stdout : undefined
      const stderr = typeof p.stderr === 'string' ? p.stderr : undefined

      const cell = cellMap.value.get(cellId)
      if (cell) {
        // The canonical output replaces any streamed partial buffer.
        cell.streamBuffer = undefined
        cell.streamAttempt = undefined
        cell.durationMs = p.duration_ms
        cell.displayOutputs = displayOutputs
        // If cell_output carries fresh stdout/stderr, overwrite.
        // Otherwise keep whatever earlier cell_console messages
        // streamed in (so the console doesn't disappear on cache
        // hits that don't re-stream).
        if (stdout !== undefined) cell.consoleStdout = stdout
        if (stderr !== undefined) cell.consoleStderr = stderr
        if (p.execution_method) {
          cell.executorName = p.execution_method
        }
        applyRemoteExecutionMetadata(cell, p)
        if (p.execution_method === 'executor') {
          updateWorkerHealth(effectiveWorkerNameForCell(cell), 'healthy')
        }
        // Capture suggest_install + language for "click to install"
        // UX. The button in CellEditor.vue only shows for ``python``;
        // an R suggestion hydrates the field so it appears in tests
        // and future install actions, but the visible button stays
        // hidden until the R install endpoint ships.
        cell.suggestInstall = p.suggest_install || undefined
        cell.suggestInstallLanguage =
          p.suggest_install_language === 'r'
            ? 'r'
            : p.suggest_install_language === 'python'
              ? 'python'
              : undefined
      }

      setCellOutput(cellId, output, displayOutputs)
    })

    wsInstance.onMessage('cell_output_delta', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const cellId = p.cell_id as CellId
      const cell = cellMap.value.get(cellId)
      if (!cell) return
      const text = typeof p.text === 'string' ? p.text : ''
      const attempt = Number(p.attempt) || 1
      if (p.kind === 'retry') {
        // Schema validation failed and the backend is retrying —
        // clear the buffer so attempt N's invalid JSON never fuses
        // with attempt N+1's corrected output.
        cell.streamBuffer = ''
        cell.streamAttempt = attempt
        return
      }
      if (cell.streamAttempt !== attempt) {
        cell.streamBuffer = ''
        cell.streamAttempt = attempt
      }
      cell.streamBuffer = (cell.streamBuffer ?? '') + text
    })

    wsInstance.onMessage('cell_iteration_progress', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const cellId = p.cell_id as CellId
      const cell = cellMap.value.get(cellId)
      if (!cell) return
      const progress: LoopProgress = {
        iteration: Number(p.iteration),
        maxIter: Number(p.max_iter),
        artifactUri: typeof p.artifact_uri === 'string' ? p.artifact_uri : undefined,
        contentType: typeof p.content_type === 'string' ? p.content_type : undefined,
        untilReached: Boolean(p.until_reached),
        iterDurationMs:
          typeof p.duration_ms === 'number' ? p.duration_ms : Number(p.duration_ms) || undefined,
      }
      cell.loopProgress = progress
    })

    wsInstance.onMessage('cell_variant_progress', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const cellId = p.cell_id as CellId
      const cell = cellMap.value.get(cellId)
      if (!cell) return
      const entry: VariantProgress = {
        variant: String(p.variant),
        index: Number(p.index),
        total: Number(p.total),
        success: Boolean(p.success),
        durationMs:
          typeof p.duration_ms === 'number' ? p.duration_ms : Number(p.duration_ms) || undefined,
        error: typeof p.error === 'string' ? p.error : undefined,
      }
      // index 0 begins a fresh fan-out run; later frames accumulate.
      const existing = entry.index === 0 ? [] : (cell.variantProgress ?? [])
      cell.variantProgress = [...existing.filter((e) => e.variant !== entry.variant), entry]
    })

    wsInstance.onMessage('cell_console', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const cellId = p.cell_id as CellId
      const text = typeof p.text === 'string' ? p.text : ''
      const stream = p.stream === 'stderr' ? 'stderr' : 'stdout'
      const cell = cellMap.value.get(cellId)
      if (cell && text) {
        // Append to the stream-specific field on the cell. Storing
        // console on the Cell (not inside output.scalar) keeps the
        // display value clean — important for cells whose output is
        // a user-controlled JSON object under @output_schema.
        if (stream === 'stderr') {
          cell.consoleStderr = (cell.consoleStderr ?? '') + text
        } else {
          cell.consoleStdout = (cell.consoleStdout ?? '') + text
        }
      }
    })

    wsInstance.onMessage('cell_error', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const cellId = p.cell_id as CellId
      const error = p.error as string
      const cell = cellMap.value.get(cellId)
      if (cell) {
        cell.status = 'error'
        cell.streamBuffer = undefined
        cell.streamAttempt = undefined
        cell.displayOutputs = []
        applyRemoteExecutionMetadata(cell, p)
        const workerName = effectiveWorkerNameForCell(cell)
        const workerEntry = availableWorkers.value.find((worker) => worker.name === workerName)
        if (workerEntry?.backend === 'executor' || error.includes('Remote executor')) {
          cell.executorName = 'executor'
          if (isRemoteExecutorLikelyUnreachable(error)) {
            updateWorkerHealth(workerName, 'unavailable')
          } else if (workerEntry?.backend === 'executor') {
            updateWorkerHealth(workerName, 'healthy')
          }
        }
        cell.output = { contentType: 'json/object', error }
        cell.suggestInstall = p.suggest_install || undefined
        cell.suggestInstallLanguage =
          p.suggest_install_language === 'r'
            ? 'r'
            : p.suggest_install_language === 'python'
              ? 'python'
              : undefined
      }
    })

    wsInstance.onMessage('cell_test_status', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const cell = cellMap.value.get(p.cell_id as CellId)
      if (cell) cell.testStatus = p.status as CellTestStatus
    })

    wsInstance.onMessage('cell_test_results', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const cell = cellMap.value.get(p.cell_id as CellId)
      if (cell) cell.testResult = parseBackendTestResult(p)
    })

    wsInstance.onMessage('dag_update', (msg: WsMessage) => {
      const dagData = msg.payload as Record<string, any>
      if (dagData.edges) {
        applyBackendDag({
          edges: dagData.edges,
          leaves: dagData.leaves,
          roots: dagData.roots,
          topological_order: dagData.topological_order,
          variant_groups: dagData.variant_groups,
        })
      }
      // Merge authoritative cell analysis from backend (defines,
      // references, upstream/downstream). Arrives asynchronously
      // after WS cell_source_update — no blocking round-trip.
      if (dagData.cells && Array.isArray(dagData.cells)) {
        for (const sc of dagData.cells) {
          const cell = cellMap.value.get(sc.id as CellId)
          if (!cell) continue
          if (Array.isArray(sc.defines)) cell.defines = sc.defines
          if (Array.isArray(sc.references)) cell.references = sc.references
          if (Array.isArray(sc.upstream_ids)) cell.upstreamIds = sc.upstream_ids
          if (Array.isArray(sc.downstream_ids)) cell.downstreamIds = sc.downstream_ids
          if (typeof sc.is_leaf === 'boolean') cell.isLeaf = sc.is_leaf
          if (Array.isArray(sc.annotation_diagnostics)) {
            cell.annotationDiagnostics = sc.annotation_diagnostics
              .filter((d: any) => d && typeof d.code === 'string')
              .map((d: any) => ({
                severity: d.severity === 'info' ? ('info' as const) : ('warn' as const),
                code: String(d.code),
                message: String(d.message ?? ''),
                line: typeof d.line === 'number' ? d.line : null,
              }))
          }
          if (typeof sc.is_module_cell === 'boolean') {
            cell.isModuleCell = sc.is_module_cell
            cell.moduleExports = Array.isArray(sc.module_exports)
              ? sc.module_exports
                  .filter((e: any) => e && typeof e.name === 'string')
                  .map((e: any) => ({ name: String(e.name), kind: String(e.kind ?? '') }))
              : undefined
          }
          // Variant fields update when the user edits a `# @variant`
          // annotation in source; for plain switches the values are
          // unchanged but still safe to overwrite.
          if (sc.variant_group !== undefined) {
            cell.variantGroup = typeof sc.variant_group === 'string' ? sc.variant_group : null
          }
          if (sc.variant_name !== undefined) {
            cell.variantName = typeof sc.variant_name === 'string' ? sc.variant_name : null
          }
          if (typeof sc.variant_active === 'boolean') {
            cell.variantActive = sc.variant_active
          }
        }
      }
    })

    wsInstance.onMessage('cascade_prompt', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const cellId = p.cell_id as CellId
      const planId = p.plan_id as string

      // Always auto-accept cascades. If the backend says upstream cells
      // need to run first, just do it. A confirmation dialog can be added
      // later for expensive re-runs; for now, eliminate friction.
      if (planId) {
        executeCascadeWebSocket(cellId, planId)
      }
    })

    wsInstance.onMessage('notebook_state', (msg: WsMessage) => {
      const state = msg.payload as Record<string, any>
      syncEnvironmentPayloadFromBackend(state)
      if ('worker' in state) {
        syncNotebookWorkerFromBackend(state.worker)
      }
      if ('workers' in state) {
        syncNotebookWorkersFromBackend(state.workers)
      }
      if ('timeout' in state) {
        syncNotebookTimeoutFromBackend(state.timeout)
      }
      if ('env' in state) {
        syncNotebookEnvFromBackend(state.env)
      }
      if ('mounts' in state) {
        syncNotebookMountsFromBackend(state.mounts)
      }
      if (state.cells && Array.isArray(state.cells)) {
        const localIds = new Set(notebook.cells.map((c) => c.id))
        const serverIds = new Set(state.cells.map((c: any) => c.id))
        const cellsAdded = state.cells.some((c: any) => !localIds.has(c.id))
        const cellsRemoved = notebook.cells.some((c) => !serverIds.has(c.id))

        if (cellsAdded || cellsRemoved) {
          // Full replace when cells were added or removed (e.g. by agent)
          notebook.cells = state.cells.map(parseBackendCellPayload)
          notebook.cells.sort((a, b) => a.order - b.order)
        } else {
          syncCellsFromBackend(state.cells)
        }
      }
      if (state.dag) {
        applyBackendDag(state.dag)
      }
    })

    wsInstance.onMessage('impact_preview', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      currentImpactPreview.value = {
        targetCellId: p.target_cell_id,
        upstream: (p.upstream || []).map((s: any) => ({
          cellId: s.cell_id,
          cellName: s.cell_name,
          skip: s.skip,
          reason: s.reason,
          estimatedMs: s.estimated_ms,
        })),
        downstream: (p.downstream || []).map((d: any) => ({
          cellId: d.cell_id,
          cellName: d.cell_name,
          currentStatus: d.current_status,
          newStatus: d.new_status,
        })),
        estimatedMs: p.estimated_ms || 0,
      }
    })

    wsInstance.onMessage('profiling_summary', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      profilingSummary.value = {
        totalExecutionMs: p.total_execution_ms || 0,
        cacheHits: p.cache_hits || 0,
        cacheMisses: p.cache_misses || 0,
        cacheSavingsMs: p.cache_savings_ms || 0,
        totalArtifactBytes: p.total_artifact_bytes || 0,
        cellProfiles: (p.cell_profiles || []).map((cp: any) => ({
          cellId: cp.cell_id,
          cellName: cp.cell_name,
          status: cp.status,
          durationMs: cp.duration_ms,
          cacheHit: cp.cache_hit,
          artifactUri: cp.artifact_uri,
          executionCount: cp.execution_count,
        })),
      }
    })

    wsInstance.onMessage('inspect_result', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const action = p.action as string
      const cellId = p.cell_id as CellId | undefined
      if (!cellId) return

      if (action === 'open') {
        // Clear the in-flight marker regardless of success.
        if (inspectPendingOpens.value.has(cellId)) {
          const next = new Set(inspectPendingOpens.value)
          next.delete(cellId)
          inspectPendingOpens.value = next
        }
        if (!inspectOpenOrder.value.includes(cellId)) {
          inspectOpenOrder.value = [...inspectOpenOrder.value, cellId]
        }
        const ready = p.ok === true
        const nextReady = new Map(inspectReadyMap.value)
        nextReady.set(cellId, ready)
        inspectReadyMap.value = nextReady
        const nextHistory = new Map(inspectHistoryMap.value)
        const entries: InspectEntry[] = []
        if (!ready) {
          entries.push({
            expr: '(open)',
            error: p.error || p.result || 'Failed to open inspect session',
          })
        }
        nextHistory.set(cellId, entries)
        inspectHistoryMap.value = nextHistory
      } else if (action === 'eval') {
        const nextHistory = new Map(inspectHistoryMap.value)
        const entries = [...(nextHistory.get(cellId) ?? [])]
        entries.push({
          expr: p.expr || '',
          result: p.ok ? p.result : undefined,
          error: p.ok ? undefined : p.error,
          type: p.type,
          stdout: p.stdout,
        })
        nextHistory.set(cellId, entries)
        inspectHistoryMap.value = nextHistory
      } else if (action === 'close') {
        inspectOpenOrder.value = inspectOpenOrder.value.filter((id) => id !== cellId)
        const nextReady = new Map(inspectReadyMap.value)
        nextReady.delete(cellId)
        inspectReadyMap.value = nextReady
        const nextHistory = new Map(inspectHistoryMap.value)
        nextHistory.delete(cellId)
        inspectHistoryMap.value = nextHistory
      }
    })

    wsInstance.onMessage('dependency_changed', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      syncEnvironmentPayloadFromBackend(p)
      if (p.cells && Array.isArray(p.cells)) {
        syncCellsFromBackend(p.cells)
      }
      if (p.action === 'add' || p.action === 'remove') {
        setEnvironmentActionSummary({
          action: p.action,
          packageName: typeof p.package === 'string' ? p.package : null,
          lockfileChanged: p.lockfile_changed === true,
          staleCellCount: Number(p.stale_cell_count ?? 0),
        })
      }
      if (!p.success && p.error) {
        dependencyError.value = p.error
      } else {
        dependencyError.value = null
        environmentError.value = null
        environmentWarnings.value = []
      }
      dependencyLoading.value = false
    })

    function handleEnvironmentJobMessage(payload: Record<string, any>) {
      syncEnvironmentPayloadFromBackend(payload)
      if (payload.cells && Array.isArray(payload.cells)) {
        syncCellsFromBackend(payload.cells)
      }

      const job = parseEnvironmentOperation(payload.environment_job)
      if (!job) return

      if (job.status === 'running') {
        dependencyError.value = null
        environmentError.value = null
        environmentWarnings.value = []
        return
      }

      dependencyLoading.value = false
      environmentLoading.value = false

      if (job.status === 'completed') {
        setEnvironmentActionSummary({
          action: job.action,
          packageName: job.packageName,
          lockfileChanged: payload.lockfile_changed === true || job.lockfileChanged,
          staleCellCount:
            typeof payload.stale_cell_count === 'number'
              ? payload.stale_cell_count
              : job.staleCellCount,
        })
        dependencyError.value = null
        environmentError.value = null
        environmentWarnings.value = Array.isArray(payload.warnings)
          ? payload.warnings.filter(
              (warning: unknown): warning is string => typeof warning === 'string',
            )
          : []
        if (job.action === 'import') {
          environmentImportPreview.value = null
        }
        // R jobs ship a fresh ``r_environment`` block in the finished
        // payload (the panel's status pill + lock hash + last sync
        // come from there), but the installed-package listing is
        // fetched on demand via a dedicated route. Trigger that
        // refresh here so the package list reflects the post-install
        // library without waiting for the next panel mount.
        if (job.action === 'r_init' || job.action === 'r_add') {
          void fetchRPackagesAction()
        }
      } else {
        const message = job.error || 'Environment update failed'
        environmentWarnings.value = Array.isArray(payload.warnings)
          ? payload.warnings.filter(
              (warning: unknown): warning is string => typeof warning === 'string',
            )
          : []
        if (job.action === 'add' || job.action === 'remove') {
          dependencyError.value = message
        } else {
          environmentError.value = message
        }
      }
    }

    wsInstance.onMessage('environment_job_started', (msg: WsMessage) => {
      handleEnvironmentJobMessage(msg.payload as Record<string, any>)
    })

    wsInstance.onMessage('environment_job_progress', (msg: WsMessage) => {
      handleEnvironmentJobMessage(msg.payload as Record<string, any>)
    })

    wsInstance.onMessage('environment_job_finished', (msg: WsMessage) => {
      handleEnvironmentJobMessage(msg.payload as Record<string, any>)
    })

    wsInstance.onMessage('error', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      if (p.code === 'ENVIRONMENT_BUSY' && typeof p.error === 'string') {
        environmentError.value = p.error
      }
    })

    wsInstance.onMessage('agent_progress', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      agentProgress.value.push({
        event: String(p.event || ''),
        detail: String(p.detail || ''),
        timestamp: Date.now(),
      })
    })

    // Live narrative chunks from the agent's intermediate turns. We
    // append them onto a streaming assistant message so the user sees
    // the agent "think out loud" instead of waiting for `agent_done`.
    wsInstance.onMessage('agent_text_delta', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      const text = String(p.text || '')
      if (!text) return
      const last = llmMessages.value.at(-1)
      if (last && last.role === 'assistant' && last.streaming && last.source === 'agent') {
        last.content += text
      } else {
        llmMessages.value.push({
          role: 'assistant',
          content: text,
          timestamp: Date.now(),
          streaming: true,
          source: 'agent',
        })
      }
    })

    // Approval prompt for a destructive tool. The user must accept or
    // decline before the agent loop continues. Auto-approve mode bypasses
    // this on the backend, so this handler only fires when the user has
    // opted into manual approval.
    wsInstance.onMessage('agent_confirm_request', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      if (typeof p.request_id !== 'string') return
      agentApprovalPrompts.value.push({
        request_id: p.request_id,
        tool: String(p.tool || ''),
        arguments: (p.arguments as Record<string, any>) ?? {},
        summary: String(p.summary || ''),
      })
    })

    wsInstance.onMessage('agent_done', (msg: WsMessage) => {
      const p = msg.payload as Record<string, any>
      _handleAgentDone(p)
    })

    wsInstance.connect()
  }
}

async function waitForWebSocket(timeoutMs: number = 5000): Promise<void> {
  if (wsInstance && wsInstance.connected()) return
  if (wsInstance) {
    await wsInstance.waitForConnection(timeoutMs)
  }
}

function cleanupWebSocket() {
  if (wsInstance) {
    wsInstance.cleanup()
    wsInstance = null
  }
}

// Session-only display selection for sweep variant groups (group → cellId).
// In sweep mode every member is "active", so the tab strip picks which member's
// source is shown in the single collapsed editor slot; switch groups ignore it.
const variantDisplaySelection = ref<Record<string, string>>({})

function variantDisplayCellId(group: VariantGroup): string {
  return variantDisplaySelection.value[group.group] || group.activeCellId
}

function selectVariantDisplay(group: string, cellId: string): void {
  variantDisplaySelection.value = { ...variantDisplaySelection.value, [group]: cellId }
}

function setVariantActive(group: string, name: string): void {
  if (!wsInstance) return
  wsInstance.setVariantActive(group, name)
}

function addVariant(group: string): void {
  if (!wsInstance) return
  wsInstance.addVariant(group)
}

function updateWidgetValues(cellId: CellId, values: Record<string, unknown>): void {
  if (!wsInstance) return
  // Optimistic local update so the control reflects the change immediately;
  // the backend re-materializes + broadcasts staleness for downstream cells.
  const cell = notebook.cells.find((c) => c.id === cellId)
  if (cell?.widget) {
    cell.widget = { ...cell.widget, values: { ...cell.widget.values, ...values } }
  }
  wsInstance.sendWidgetUpdate(cellId, values)
}

async function executeCellWebSocket(cellId: CellId) {
  if (environmentMutationActive.value) {
    environmentError.value =
      environmentOperation.value?.error ||
      'Environment update in progress. Running cells is disabled until it finishes.'
    return
  }
  if (!wsInstance) {
    console.warn('[notebook] No WebSocket instance, cannot execute cell:', cellId)
    return
  }
  if (!wsInstance.connected()) {
    try {
      await wsInstance.waitForConnection(5000)
    } catch {
      console.warn('[notebook] WebSocket not connected, cannot execute cell:', cellId)
      return
    }
  }
  // Flush any pending source edits so the backend has the latest
  // source before execution begins.
  flushCellSource(cellId)

  // Clear any stale loop progress so the badge restarts from zero.
  const cell = cellMap.value.get(cellId)
  if (cell?.loopProgress) {
    cell.loopProgress = undefined
  }
  if (cell) {
    cell.streamBuffer = undefined
    cell.streamAttempt = undefined
  }

  setCellStatus(cellId, 'running')
  wsInstance.executeCell(cellId)
}

async function executeNotebookRunAllWebSocket() {
  if (environmentMutationActive.value) {
    environmentError.value =
      environmentOperation.value?.error ||
      'Environment update in progress. Running cells is disabled until it finishes.'
    return
  }
  if (!wsInstance) {
    console.warn('[notebook] No WebSocket instance, cannot run all cells')
    return
  }
  if (!wsInstance.connected()) {
    try {
      await wsInstance.waitForConnection(5000)
    } catch {
      console.warn('[notebook] WebSocket not connected, cannot run all cells')
      return
    }
  }
  // Flush every dirty cell first — otherwise run-all hits the backend with
  // the stale debounced copy of any cell the user just edited.
  flushDirtyCells()
  wsInstance.executeNotebookRunAll()
}

function executeCascadeWebSocket(cellId: CellId, planId: string) {
  if (wsInstance && wsInstance.connected()) {
    wsInstance.executeCascade(cellId, planId)
  }
}

async function executeForceWebSocket(cellId: CellId) {
  if (environmentMutationActive.value) {
    environmentError.value =
      environmentOperation.value?.error ||
      'Environment update in progress. Running cells is disabled until it finishes.'
    return
  }
  if (!wsInstance) {
    console.warn('[notebook] No WebSocket instance, cannot force cell:', cellId)
    return
  }
  if (!wsInstance.connected()) {
    try {
      await wsInstance.waitForConnection(5000)
    } catch {
      console.warn('[notebook] WebSocket not connected, cannot force cell:', cellId)
      return
    }
  }
  // "Run this only" runs the target with whatever upstream artifacts are
  // already on disk, so flushing the target's source is enough — we
  // explicitly do not want to flush upstreams here.
  flushCellSource(cellId)
  setCellStatus(cellId, 'running')
  wsInstance.executeForce(cellId)
}

async function executeRerunWebSocket(cellId: CellId) {
  if (environmentMutationActive.value) {
    environmentError.value =
      environmentOperation.value?.error ||
      'Environment update in progress. Running cells is disabled until it finishes.'
    return
  }
  if (!wsInstance) {
    console.warn('[notebook] No WebSocket instance, cannot rerun cell:', cellId)
    return
  }
  if (!wsInstance.connected()) {
    try {
      await wsInstance.waitForConnection(5000)
    } catch {
      console.warn('[notebook] WebSocket not connected, cannot rerun cell:', cellId)
      return
    }
  }
  // Flush every dirty cell — the backend's cascade planner needs current
  // source for *all* upstreams to make the right cascade decision, not just
  // the target. A dirty upstream would otherwise look ready and get skipped.
  flushDirtyCells()
  const cell = cellMap.value.get(cellId)
  if (cell?.loopProgress) {
    cell.loopProgress = undefined
  }
  if (cell) {
    cell.streamBuffer = undefined
    cell.streamAttempt = undefined
  }
  setCellStatus(cellId, 'running')
  wsInstance.executeRerun(cellId)
}

async function executeNotebookRerunAllWebSocket() {
  if (environmentMutationActive.value) {
    environmentError.value =
      environmentOperation.value?.error ||
      'Environment update in progress. Running cells is disabled until it finishes.'
    return
  }
  if (!wsInstance) {
    console.warn('[notebook] No WebSocket instance, cannot rerun all cells')
    return
  }
  if (!wsInstance.connected()) {
    try {
      await wsInstance.waitForConnection(5000)
    } catch {
      console.warn('[notebook] WebSocket not connected, cannot rerun all cells')
      return
    }
  }
  // Flush every dirty cell first — otherwise rerun-all hits the backend with
  // the stale debounced copy of any cell the user just edited.
  flushDirtyCells()
  wsInstance.executeNotebookRerunAll()
}

function cancelCellWebSocket(cellId: CellId) {
  if (wsInstance && wsInstance.connected()) {
    wsInstance.cancelCell(cellId)
  }
}

async function fetchDependencies() {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  try {
    const data = await strata.listDependencies(sid)
    syncEnvironmentPayloadFromBackend(data)
    dependencyError.value = null
  } catch (err) {
    console.error('Failed to fetch dependencies:', err)
    dependencyError.value = err instanceof Error ? err.message : 'Failed to fetch dependencies'
  }
}

async function fetchEnvironment() {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  environmentLoading.value = true
  try {
    const data = await strata.getEnvironmentStatus(sid)
    syncEnvironmentPayloadFromBackend(data)
    environmentError.value = null
    environmentWarnings.value = []
  } catch (err) {
    console.error('Failed to fetch environment:', err)
    environmentError.value = err instanceof Error ? err.message : 'Failed to load environment'
  } finally {
    environmentLoading.value = false
  }
}

async function fetchWorkers(forceRefresh = false) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  workerHealthLoading.value = true
  try {
    const data = await strata.listWorkers(sid, forceRefresh)
    if (data.workers && Array.isArray(data.workers)) {
      syncWorkerCatalogFromBackend(data.workers)
    }
    if ('definitions_editable' in data) {
      syncWorkerDefinitionsEditableFromBackend(data.definitions_editable)
    }
    syncWorkerHealthCheckedAtFromBackend(data.health_checked_at)
    if (data.definitions_editable === false) {
      await fetchServerWorkerRegistry(forceRefresh)
    } else {
      clearServerWorkerRegistryState()
    }
  } catch (err) {
    console.error('Failed to fetch workers:', err)
  } finally {
    workerHealthLoading.value = false
  }
}

async function ensureWorkersLoaded(forceRefresh = false) {
  if (!forceRefresh && (workerCatalogLoaded.value || workerHealthLoading.value)) {
    return
  }
  await fetchWorkers(forceRefresh)
}

async function fetchServerWorkerRegistry(forceRefresh = false) {
  const strata = useStrata()
  serverWorkerRegistryLoading.value = true
  try {
    const data = await strata.listAdminNotebookWorkers(forceRefresh)
    syncServerManagedWorkersFromBackend(data.configured_workers)
    serverWorkerRegistryAvailable.value = true
    serverWorkerRegistryError.value = null
  } catch (err) {
    serverManagedWorkers.value = []
    serverWorkerRegistryAvailable.value = false
    serverWorkerRegistryError.value = null
    console.debug('Admin notebook worker registry unavailable:', err)
  } finally {
    serverWorkerRegistryLoading.value = false
  }
}

function setServerWorkerActionLoading(workerName: string, loading: boolean) {
  if (loading) {
    serverWorkerActionLoading.value = {
      ...serverWorkerActionLoading.value,
      [workerName]: true,
    }
    return
  }

  const nextLoading = { ...serverWorkerActionLoading.value }
  delete nextLoading[workerName]
  serverWorkerActionLoading.value = nextLoading
}

function isServerWorkerActionLoading(workerName: string): boolean {
  return serverWorkerActionLoading.value[workerName] === true
}

async function updatePythonVersionAction(
  newVersion: string,
): Promise<{ accepted: boolean; error?: string }> {
  const sid = sessionId()
  if (!sid) return { accepted: false, error: 'No active session' }
  const strata = useStrata()
  try {
    const data = await strata.updateNotebookPythonVersion(sid, newVersion)
    // 200 no-op + 202 accepted both surface as no thrown error. The
    // environment-job WS broadcasts handle live progress updates, so
    // we don't have to refresh the panel inline here.
    return { accepted: data.accepted }
  } catch (err: any) {
    return { accepted: false, error: err?.message || 'Failed to update Python version' }
  }
}

async function addDependencyAction(pkg: string) {
  const sid = sessionId()
  if (!sid) return
  dependencyLoading.value = true
  dependencyError.value = null
  environmentError.value = null
  environmentWarnings.value = []
  environmentImportPreview.value = null
  beginEnvironmentOperation('add', `uv add ${pkg}`, pkg)
  const strata = useStrata()
  try {
    const data = await strata.addDependency(sid, pkg)
    syncEnvironmentPayloadFromBackend(data)
    if (data.cells && Array.isArray(data.cells)) {
      syncCellsFromBackend(data.cells)
    }
  } catch (err: any) {
    dependencyError.value = err.message || 'Failed to add dependency'
    syncEnvironmentPayloadFromBackend(err?.payload)
    finishEnvironmentOperation(
      'add',
      'failed',
      extractOperationLogPayload(err?.payload),
      `uv add ${pkg}`,
    )
  }
}

async function initRenvAction() {
  const sid = sessionId()
  if (!sid) return
  dependencyError.value = null
  environmentError.value = null
  environmentWarnings.value = []
  environmentImportPreview.value = null
  const command = "Rscript -e 'renv::init(bare = TRUE)'"
  // Reuse the env-operation block for visual parity with uv jobs —
  // the action enum has been widened (see types/notebook.ts) but
  // the UI render is unchanged.
  beginEnvironmentOperation('r_init' as any, command, null)
  const strata = useStrata()
  try {
    const data = await strata.initRenv(sid)
    syncEnvironmentPayloadFromBackend(data)
    if (data.cells && Array.isArray(data.cells)) {
      syncCellsFromBackend(data.cells)
    }
    // The 202 response just acknowledges the job submission; the
    // background job is still running. ``handleEnvironmentJobMessage``
    // fires ``fetchRPackagesAction`` when the WS finished frame
    // lands with a completed status — no eager fetch here, which
    // would race against the still-running subprocess and pull
    // pre-install package state.
  } catch (err: any) {
    dependencyError.value = err.message || 'Failed to initialise renv'
    syncEnvironmentPayloadFromBackend(err?.payload)
    finishEnvironmentOperation(
      'r_init' as any,
      'failed',
      extractOperationLogPayload(err?.payload),
      command,
    )
  }
}

async function addRPackageAction(pkg: string) {
  const sid = sessionId()
  if (!sid) return
  dependencyError.value = null
  environmentError.value = null
  environmentWarnings.value = []
  environmentImportPreview.value = null
  const command = `Rscript -e 'renv::install("${pkg}"); renv::snapshot(type = "all", prompt = FALSE)'`
  beginEnvironmentOperation('r_add' as any, command, pkg)
  const strata = useStrata()
  try {
    const data = await strata.addRPackage(sid, pkg)
    syncEnvironmentPayloadFromBackend(data)
    if (data.cells && Array.isArray(data.cells)) {
      syncCellsFromBackend(data.cells)
    }
    // 202 only. WS ``environment_job_finished`` triggers the
    // package-list refresh once the job actually completes — see
    // ``handleEnvironmentJobMessage``.
  } catch (err: any) {
    dependencyError.value = err.message || `Failed to install R package ${pkg}`
    syncEnvironmentPayloadFromBackend(err?.payload)
    finishEnvironmentOperation(
      'r_add' as any,
      'failed',
      extractOperationLogPayload(err?.payload),
      command,
    )
  }
}

async function removeDependencyAction(pkg: string) {
  const sid = sessionId()
  if (!sid) return
  dependencyLoading.value = true
  dependencyError.value = null
  environmentError.value = null
  environmentWarnings.value = []
  environmentImportPreview.value = null
  beginEnvironmentOperation('remove', `uv remove ${pkg}`, pkg)
  const strata = useStrata()
  try {
    const data = await strata.removeDependency(sid, pkg)
    syncEnvironmentPayloadFromBackend(data)
    if (data.cells && Array.isArray(data.cells)) {
      syncCellsFromBackend(data.cells)
    }
  } catch (err: any) {
    dependencyError.value = err.message || 'Failed to remove dependency'
    syncEnvironmentPayloadFromBackend(err?.payload)
    finishEnvironmentOperation(
      'remove',
      'failed',
      extractOperationLogPayload(err?.payload),
      `uv remove ${pkg}`,
    )
  }
}

async function syncEnvironmentAction() {
  const sid = sessionId()
  if (!sid) return
  environmentLoading.value = true
  environmentError.value = null
  environmentWarnings.value = []
  environmentImportPreview.value = null
  beginEnvironmentOperation('sync', 'uv sync')
  const strata = useStrata()
  try {
    const data = await strata.syncEnvironment(sid)
    syncEnvironmentPayloadFromBackend(data)
    if (data.cells && Array.isArray(data.cells)) {
      syncCellsFromBackend(data.cells)
    }
  } catch (err: any) {
    environmentError.value = err.message || 'Failed to sync environment'
    syncEnvironmentPayloadFromBackend(err?.payload)
    finishEnvironmentOperation(
      'sync',
      'failed',
      extractOperationLogPayload(err?.payload),
      'uv sync',
    )
  }
}

async function exportRequirementsAction(): Promise<string> {
  const sid = sessionId()
  if (!sid) return ''
  environmentLoading.value = true
  environmentError.value = null
  environmentWarnings.value = []
  const strata = useStrata()
  try {
    return await strata.exportRequirements(sid)
  } catch (err: any) {
    environmentError.value = err.message || 'Failed to export requirements.txt'
    return ''
  } finally {
    environmentLoading.value = false
  }
}

async function previewRequirementsImportAction(requirements: string) {
  const sid = sessionId()
  if (!sid) return
  environmentLoading.value = true
  environmentError.value = null
  environmentWarnings.value = []
  const strata = useStrata()
  try {
    const data = await strata.previewRequirementsImport(sid, requirements)
    syncEnvironmentPayloadFromBackend(data)
    environmentWarnings.value = Array.isArray(data.warnings)
      ? data.warnings.filter((warning: unknown): warning is string => typeof warning === 'string')
      : []
    environmentImportPreview.value = {
      kind: 'requirements',
      previewDependencies: Array.isArray(data.preview_dependencies)
        ? data.preview_dependencies.map((dep: any) => parseDependencyInfo(dep))
        : [],
      normalizedRequirements: Array.isArray(data.normalized_requirements)
        ? data.normalized_requirements.filter(
            (entry: unknown): entry is string => typeof entry === 'string',
          )
        : [],
      importedCount: Number(data.imported_count ?? 0),
      warnings: environmentWarnings.value,
      additions: Array.isArray(data.additions)
        ? data.additions.map((dep: any) => parseDependencyInfo(dep))
        : [],
      removals: Array.isArray(data.removals)
        ? data.removals.map((dep: any) => parseDependencyInfo(dep))
        : [],
      unchanged: Array.isArray(data.unchanged)
        ? data.unchanged.map((dep: any) => parseDependencyInfo(dep))
        : [],
    }
  } catch (err: any) {
    environmentImportPreview.value = null
    environmentError.value = err.message || 'Failed to preview requirements.txt import'
  } finally {
    environmentLoading.value = false
  }
}

async function importRequirementsAction(requirements: string) {
  const sid = sessionId()
  if (!sid) return
  environmentLoading.value = true
  environmentError.value = null
  environmentWarnings.value = []
  beginEnvironmentOperation('import', 'uv sync')
  const strata = useStrata()
  try {
    const data = await strata.importRequirements(sid, requirements)
    syncEnvironmentPayloadFromBackend(data)
    if (data.cells && Array.isArray(data.cells)) {
      syncCellsFromBackend(data.cells)
    }
  } catch (err: any) {
    environmentError.value = err.message || 'Failed to import requirements.txt'
    syncEnvironmentPayloadFromBackend(err?.payload)
    finishEnvironmentOperation(
      'import',
      'failed',
      extractOperationLogPayload(err?.payload),
      'uv sync',
    )
  }
}

async function importEnvironmentYamlAction(environmentYaml: string) {
  const sid = sessionId()
  if (!sid) return
  environmentLoading.value = true
  environmentError.value = null
  environmentWarnings.value = []
  beginEnvironmentOperation('import', 'uv sync')
  const strata = useStrata()
  try {
    const data = await strata.importEnvironmentYaml(sid, environmentYaml)
    syncEnvironmentPayloadFromBackend(data)
    if (data.cells && Array.isArray(data.cells)) {
      syncCellsFromBackend(data.cells)
    }
  } catch (err: any) {
    environmentError.value = err.message || 'Failed to import environment.yaml'
    syncEnvironmentPayloadFromBackend(err?.payload)
    finishEnvironmentOperation(
      'import',
      'failed',
      extractOperationLogPayload(err?.payload),
      'uv sync',
    )
  }
}

async function previewEnvironmentYamlImportAction(environmentYaml: string) {
  const sid = sessionId()
  if (!sid) return
  environmentLoading.value = true
  environmentError.value = null
  environmentWarnings.value = []
  const strata = useStrata()
  try {
    const data = await strata.previewEnvironmentYamlImport(sid, environmentYaml)
    syncEnvironmentPayloadFromBackend(data)
    environmentWarnings.value = Array.isArray(data.warnings)
      ? data.warnings.filter((warning: unknown): warning is string => typeof warning === 'string')
      : []
    environmentImportPreview.value = {
      kind: 'environment_yaml',
      previewDependencies: Array.isArray(data.preview_dependencies)
        ? data.preview_dependencies.map((dep: any) => parseDependencyInfo(dep))
        : [],
      normalizedRequirements: Array.isArray(data.normalized_requirements)
        ? data.normalized_requirements.filter(
            (entry: unknown): entry is string => typeof entry === 'string',
          )
        : [],
      importedCount: Number(data.imported_count ?? 0),
      warnings: environmentWarnings.value,
      additions: Array.isArray(data.additions)
        ? data.additions.map((dep: any) => parseDependencyInfo(dep))
        : [],
      removals: Array.isArray(data.removals)
        ? data.removals.map((dep: any) => parseDependencyInfo(dep))
        : [],
      unchanged: Array.isArray(data.unchanged)
        ? data.unchanged.map((dep: any) => parseDependencyInfo(dep))
        : [],
    }
  } catch (err: any) {
    environmentImportPreview.value = null
    environmentError.value = err.message || 'Failed to preview environment.yaml import'
  } finally {
    environmentLoading.value = false
  }
}

function clearEnvironmentImportPreview() {
  environmentImportPreview.value = null
}

async function updateNotebookMountsAction(mounts: MountSpec[]) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  const data = await strata.updateNotebookMounts(sid, mounts)
  if (data.mounts) {
    syncNotebookMountsFromBackend(data.mounts)
  }
  if (data.cells && Array.isArray(data.cells)) {
    syncCellsFromBackend(data.cells)
  }
}

async function updateNotebookConnectionsAction(connections: ConnectionSpec[]) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  const data = await strata.updateNotebookConnections(sid, connections)
  notebook.connections = Array.isArray(data.connections) ? data.connections : []
}

async function getConnectionSchemaAction(name: string) {
  const sid = sessionId()
  if (!sid) {
    throw new Error("Notebook session isn't open yet")
  }
  const strata = useStrata()
  return strata.getConnectionSchema(sid, name)
}

async function updateNotebookWorkerAction(worker: string | null) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  notebookWorkerError.value = null
  try {
    const data = await strata.updateNotebookWorker(sid, worker)
    if ('worker' in data) {
      syncNotebookWorkerFromBackend(data.worker)
    }
    if (data.cells && Array.isArray(data.cells)) {
      syncCellsFromBackend(data.cells)
    }
    if (data.workers && Array.isArray(data.workers)) {
      syncWorkerCatalogFromBackend(data.workers)
      syncWorkerHealthCheckedAtFromBackend(data.health_checked_at)
    } else {
      fetchWorkers()
    }
    if ('definitions_editable' in data) {
      syncWorkerDefinitionsEditableFromBackend(data.definitions_editable)
    }
  } catch (err: any) {
    notebookWorkerError.value = err.message || 'Failed to update notebook worker'
  }
}

async function updateNotebookWorkersAction(workers: EditableWorkerSpec[]) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  workerRegistryError.value = null
  const payload = workers
    .map((worker) => ({
      name: worker.name.trim(),
      backend: worker.backend,
      runtime_id: worker.runtimeId || null,
      config: worker.config || {},
    }))
    .filter((worker) => worker.name)
  try {
    const data = await strata.updateWorkers(sid, payload)
    if (data.configured_workers && Array.isArray(data.configured_workers)) {
      syncNotebookWorkersFromBackend(data.configured_workers)
    }
    if (data.workers && Array.isArray(data.workers)) {
      syncWorkerCatalogFromBackend(data.workers)
      syncWorkerHealthCheckedAtFromBackend(data.health_checked_at)
    }
    if ('definitions_editable' in data) {
      syncWorkerDefinitionsEditableFromBackend(data.definitions_editable)
    }
  } catch (err: any) {
    workerRegistryError.value = err.message || 'Failed to update worker registry'
  }
}

async function updateServerWorkerRegistryAction(workers: EditableWorkerSpec[]) {
  const strata = useStrata()
  serverWorkerRegistryError.value = null
  const payload = workers
    .map((worker) => ({
      name: worker.name.trim(),
      backend: worker.backend,
      runtime_id: worker.runtimeId || null,
      config: worker.config || {},
      enabled: worker.enabled !== false,
    }))
    .filter((worker) => worker.name)
  try {
    const data = await strata.updateAdminNotebookWorkers(payload)
    if (data.configured_workers && Array.isArray(data.configured_workers)) {
      syncServerManagedWorkersFromBackend(data.configured_workers)
    }
    serverWorkerRegistryAvailable.value = true
    syncWorkerHealthCheckedAtFromBackend(data.health_checked_at)
    await fetchWorkers(true)
  } catch (err: any) {
    serverWorkerRegistryError.value = err.message || 'Failed to update server worker registry'
  }
}

function serializeEditableWorker(worker: EditableWorkerSpec) {
  return {
    name: worker.name.trim(),
    backend: worker.backend,
    runtime_id: worker.runtimeId || null,
    config: worker.config || {},
    enabled: worker.enabled !== false,
  }
}

async function saveServerWorkerAction(worker: EditableWorkerSpec, originalName: string | null) {
  const strata = useStrata()
  const payload = serializeEditableWorker(worker)
  if (!payload.name) {
    serverWorkerRegistryError.value = 'Worker name is required'
    return
  }

  serverWorkerRegistryError.value = null
  serverWorkerRegistryLoading.value = true
  try {
    const data = originalName
      ? await strata.replaceAdminNotebookWorker(originalName, payload)
      : await strata.createAdminNotebookWorker(payload)
    if (data.configured_workers && Array.isArray(data.configured_workers)) {
      syncServerManagedWorkersFromBackend(data.configured_workers)
    }
    serverWorkerRegistryAvailable.value = true
    syncWorkerHealthCheckedAtFromBackend(data.health_checked_at)
    await fetchWorkers(true)
  } catch (err: any) {
    serverWorkerRegistryError.value =
      err.message ||
      (originalName
        ? `Failed to update worker "${originalName}"`
        : `Failed to create worker "${payload.name}"`)
  } finally {
    serverWorkerRegistryLoading.value = false
  }
}

async function deleteServerWorkerAction(workerName: string) {
  const strata = useStrata()
  serverWorkerRegistryError.value = null
  serverWorkerRegistryLoading.value = true
  try {
    const data = await strata.deleteAdminNotebookWorker(workerName)
    if (data.configured_workers && Array.isArray(data.configured_workers)) {
      syncServerManagedWorkersFromBackend(data.configured_workers)
    }
    serverWorkerRegistryAvailable.value = true
    syncWorkerHealthCheckedAtFromBackend(data.health_checked_at)
    await fetchWorkers(true)
  } catch (err: any) {
    serverWorkerRegistryError.value = err.message || `Failed to delete worker "${workerName}"`
  } finally {
    serverWorkerRegistryLoading.value = false
  }
}

async function updateServerWorkerEnabledAction(workerName: string, enabled: boolean) {
  const strata = useStrata()
  serverWorkerRegistryError.value = null
  setServerWorkerActionLoading(workerName, true)
  try {
    const data = await strata.patchAdminNotebookWorker(workerName, enabled)
    if (data.configured_workers && Array.isArray(data.configured_workers)) {
      syncServerManagedWorkersFromBackend(data.configured_workers)
    }
    serverWorkerRegistryAvailable.value = true
    syncWorkerHealthCheckedAtFromBackend(data.health_checked_at)
    await fetchWorkers(true)
  } catch (err: any) {
    serverWorkerRegistryError.value = err.message || `Failed to update worker "${workerName}"`
  } finally {
    setServerWorkerActionLoading(workerName, false)
  }
}

async function refreshServerWorkerAction(workerName: string) {
  const strata = useStrata()
  serverWorkerRegistryError.value = null
  setServerWorkerActionLoading(workerName, true)
  try {
    const data = await strata.refreshAdminNotebookWorker(workerName)
    if (data.configured_workers && Array.isArray(data.configured_workers)) {
      syncServerManagedWorkersFromBackend(data.configured_workers)
    }
    serverWorkerRegistryAvailable.value = true
    syncWorkerHealthCheckedAtFromBackend(data.health_checked_at)
    await fetchWorkers(true)
  } catch (err: any) {
    serverWorkerRegistryError.value = err.message || `Failed to refresh worker "${workerName}"`
  } finally {
    setServerWorkerActionLoading(workerName, false)
  }
}

async function updateNotebookTimeoutAction(timeout: number | null) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  const data = await strata.updateNotebookTimeout(sid, timeout)
  if ('timeout' in data) {
    syncNotebookTimeoutFromBackend(data.timeout)
  }
  if (data.cells && Array.isArray(data.cells)) {
    syncCellsFromBackend(data.cells)
  }
}

async function updateNotebookEnvAction(env: Record<string, string>) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  const data = await strata.updateNotebookEnv(sid, env)
  if ('env' in data) {
    syncNotebookEnvFromBackend(data.env)
  }
  applyEnvSources(data)
  if (data.cells && Array.isArray(data.cells)) {
    syncCellsFromBackend(data.cells)
  }
  // Re-check LLM availability — the user may have just added or
  // removed an API key, which changes whether the AI panel is active.
  void checkLlmStatus()
}

async function refreshSecretManagerAction() {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  const data = await strata.refreshNotebookSecretManager(sid)
  if ('env' in data) {
    syncNotebookEnvFromBackend(data.env)
  }
  applyEnvSources(data)
  if (data.cells && Array.isArray(data.cells)) {
    syncCellsFromBackend(data.cells)
  }
  void checkLlmStatus()
  return data
}

async function updateNotebookSecretManagerConfigAction(
  config: Record<string, string | null | undefined>,
) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  const data = await strata.updateNotebookSecretManagerConfig(sid, config)
  if ('env' in data) {
    syncNotebookEnvFromBackend(data.env)
  }
  applyEnvSources(data)
  if (data.cells && Array.isArray(data.cells)) {
    syncCellsFromBackend(data.cells)
  }
  void checkLlmStatus()
  return data
}

function updateSourceWebSocket(cellId: CellId, source: string) {
  if (wsInstance && wsInstance.connected()) {
    wsInstance.updateCellSource(cellId, source)
  }
}

function requestImpactPreview(cellId: CellId) {
  if (wsInstance && wsInstance.connected()) {
    wsInstance.send('impact_preview_request', { cell_id: cellId })
  }
}

function clearImpactPreview() {
  currentImpactPreview.value = null
}

function requestProfilingSummary() {
  if (wsInstance && wsInstance.connected()) {
    wsInstance.send('profiling_request', {})
  }
}

function openInspect(cellId: CellId) {
  if (!wsInstance || !wsInstance.connected()) return
  // Already open or mid-open for this cell? Nothing to do. (Toggle is
  // the CellEditor's job — it closes on a second click rather than
  // re-opening.)
  if (inspectOpenOrder.value.includes(cellId)) return
  if (inspectPendingOpens.value.has(cellId)) return
  if (inspectActiveCount() >= maxInspectPanels.value) {
    // At the user-configured panel cap. Ask the user to confirm closing
    // the oldest panel before we open a new one. The actual swap happens
    // in confirmInspectSwap() once the user says yes. Evict the oldest
    // *confirmed-open* panel, not a pending one — pending opens will
    // resolve on their own and don't represent a user-visible panel yet.
    const evictCellId = inspectOpenOrder.value[0] ?? Array.from(inspectPendingOpens.value)[0]
    if (!evictCellId) return
    pendingInspectRequest.value = { newCellId: cellId, evictCellId }
    return
  }
  inspectPendingOpens.value = new Set([...inspectPendingOpens.value, cellId])
  wsInstance.inspectOpen(cellId)
}

function confirmInspectSwap() {
  const req = pendingInspectRequest.value
  if (!req || !wsInstance || !wsInstance.connected()) {
    pendingInspectRequest.value = null
    return
  }
  wsInstance.inspectClose(req.evictCellId)
  inspectPendingOpens.value = new Set([...inspectPendingOpens.value, req.newCellId])
  wsInstance.inspectOpen(req.newCellId)
  pendingInspectRequest.value = null
}

function cancelInspectSwap() {
  pendingInspectRequest.value = null
}

function evalInspect(cellId: CellId, expr: string) {
  if (wsInstance && wsInstance.connected() && inspectOpenOrder.value.includes(cellId)) {
    wsInstance.inspectEval(cellId, expr)
  }
}

function closeInspect(cellId: CellId) {
  if (wsInstance && wsInstance.connected() && inspectOpenOrder.value.includes(cellId)) {
    wsInstance.inspectClose(cellId)
  }
}

// --- LLM assistant ---------------------------------------------------------

async function checkLlmStatus() {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  try {
    const status = await strata.getLlmStatus(sid)
    llmAvailable.value = status.available
    llmModel.value = status.model ?? null
    llmProvider.value = status.provider ?? null
  } catch {
    llmAvailable.value = false
  }
}

/**
 * Send a chat message and stream the assistant response.
 *
 * Sends the last LLM_HISTORY_LIMIT prior turns as conversation history so
 * the LLM has context across turns within this session. The assistant
 * message is appended immediately with streaming=true and deltas are
 * written into it until the stream ends.
 */
async function llmChat(message: string, cellId?: string) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()

  // Snapshot history BEFORE appending the new user message — the new
  // message will be sent separately via the `message` field.
  const history = llmMessages.value
    .filter((m) => !m.streaming && m.content)
    .slice(-LLM_HISTORY_LIMIT)
    .map((m) => ({ role: m.role, content: m.content }))

  llmMessages.value.push({
    role: 'user',
    content: message,
    timestamp: Date.now(),
    source: 'chat',
  })

  const assistantMsg: LlmMessage = {
    role: 'assistant',
    content: '',
    timestamp: Date.now(),
    streaming: true,
    source: 'chat',
  }
  llmMessages.value.push(assistantMsg)

  llmLoading.value = true
  llmError.value = null

  try {
    for await (const event of strata.llmChatStream(sid, message, history, cellId)) {
      if (event.type === 'delta') {
        assistantMsg.content += event.text
      } else if (event.type === 'done') {
        assistantMsg.model = event.model ?? undefined
        assistantMsg.tokens = event.tokens
        assistantMsg.streaming = false
      } else if (event.type === 'error') {
        llmError.value = event.message
        assistantMsg.streaming = false
        if (!assistantMsg.content) {
          // Drop the empty placeholder so the error stands alone
          llmMessages.value = llmMessages.value.filter((m) => m !== assistantMsg)
        }
        break
      }
    }
  } catch (err: any) {
    llmError.value = err?.message || 'LLM request failed'
    assistantMsg.streaming = false
    if (!assistantMsg.content) {
      llmMessages.value = llmMessages.value.filter((m) => m !== assistantMsg)
    }
  } finally {
    assistantMsg.streaming = false
    llmLoading.value = false
  }
}

function clearLlmHistory() {
  llmMessages.value = []
  llmError.value = null
}

async function runAgentAction(message: string) {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()

  agentRunning.value = true
  agentProgress.value = []
  agentApprovalPrompts.value = []
  agentError.value = null

  llmMessages.value.push({
    role: 'user',
    content: message,
    timestamp: Date.now(),
    source: 'agent',
  })

  try {
    const resp = await strata.agentRun(sid, message, {
      autoApprove: agentAutoApprove.value,
    })
    // Agent now runs as a background task — resp just has { job_id, status }
    // Completion arrives via WS agent_done message
    if (resp.error) {
      agentError.value = resp.error
      agentRunning.value = false
    }
  } catch (err: any) {
    agentError.value = err.message || 'Agent failed to start'
    agentRunning.value = false
  }
}

function _handleAgentDone(payload: Record<string, any>) {
  agentRunning.value = false
  agentApprovalPrompts.value = []

  const content = payload.content || ''
  const error = payload.error || null

  // Finalize the streaming assistant message if one is in flight, or
  // append the final content as a new message if streaming wasn't used.
  const last = llmMessages.value.at(-1)
  if (last && last.role === 'assistant' && last.streaming && last.source === 'agent') {
    last.streaming = false
    if (content && content !== last.content) {
      // Backend may include richer final content than the streamed
      // narrative — prefer it over the partial when they differ.
      last.content = content
    }
    last.model = payload.model ?? last.model
    if (payload.tokens) {
      last.tokens = {
        input: payload.tokens.input || 0,
        output: payload.tokens.output || 0,
      }
    }
  } else if (content) {
    llmMessages.value.push({
      role: 'assistant',
      content,
      model: payload.model,
      tokens: payload.tokens
        ? { input: payload.tokens.input || 0, output: payload.tokens.output || 0 }
        : undefined,
      timestamp: Date.now(),
      source: 'agent',
    })
  }

  if (error) {
    agentError.value = error
  }

  // Refresh dependencies and environment in case agent installed packages
  fetchDependencies()
  fetchEnvironment()
}

function cancelAgent() {
  if (wsInstance && wsInstance.connected()) {
    wsInstance.send('agent_cancel', {})
  }
}

function respondToApproval(request_id: string, approved: boolean) {
  agentApprovalPrompts.value = agentApprovalPrompts.value.filter((p) => p.request_id !== request_id)
  if (wsInstance && wsInstance.connected()) {
    wsInstance.send('agent_confirm_response', { request_id, approved })
  }
}

async function resetAgentMemory() {
  const sid = sessionId()
  if (!sid) return
  const strata = useStrata()
  try {
    await strata.agentReset(sid)
    // Drop any prior agent messages from the visible thread so the user
    // sees the reset reflected immediately.
    llmMessages.value = llmMessages.value.filter((m) => m.source !== 'agent')
    agentProgress.value = []
    agentApprovalPrompts.value = []
    agentError.value = null
  } catch (err: any) {
    agentError.value = err.message || 'Reset failed'
  }
}

async function insertLlmCodeAsCell(code: string, afterCellId?: string) {
  await addCell(afterCellId)
  const sorted = orderedCells.value
  let newCell: Cell | undefined
  if (afterCellId) {
    const idx = sorted.findIndex((c) => c.id === afterCellId)
    newCell = sorted[idx + 1]
  } else {
    newCell = sorted[sorted.length - 1]
  }
  if (newCell) {
    await updateSource(newCell.id, code)
  }
}

async function insertLlmCodeAsCells(codes: string[], afterCellId?: string) {
  let insertAfter = afterCellId
  for (const code of codes) {
    await addCell(insertAfter)
    const sorted = orderedCells.value
    let newCell: Cell | undefined
    if (insertAfter) {
      const idx = sorted.findIndex((c) => c.id === insertAfter)
      newCell = sorted[idx + 1]
    } else {
      newCell = sorted[sorted.length - 1]
    }
    if (newCell) {
      await updateSource(newCell.id, code)
      insertAfter = newCell.id
    }
  }
}

// --- Public API ------------------------------------------------------------

export function useNotebook() {
  return {
    notebook,
    orderedCells,
    dagEdges,
    cellMap,
    connected,
    connectError,
    // Lifecycle
    boot,
    openNotebook,
    // Registry / dashboard (P3)
    registryArtifactsByCell,
    registryNames,
    registryPending,
    registryAudit,
    registryLoading,
    registryError,
    toasts,
    pushToast,
    dismissToast,
    refreshRegistryAction,
    fetchRegistryAuditAction,
    setAliasAction,
    approvePendingAction,
    rejectPendingAction,
    fetchLineageAction,
    lineageOpen,
    lineageTitle,
    lineageRows,
    lineageLoading,
    lineageError,
    openLineageAction,
    closeLineage,
    openBySessionId,
    updateNotebookNameAction,
    deleteNotebookAction,
    // Cell operations (always backend-backed)
    addCell,
    removeCell,
    updateSource,
    flushDirtyCells,
    flushCellSource,
    setCellStatus,
    setCellOutput,
    moveCell,
    duplicateCell,
    // WebSocket
    cleanupWebSocket,
    executeCellWebSocket,
    executeNotebookRunAllWebSocket,
    executeCascadeWebSocket,
    executeForceWebSocket,
    executeRerunWebSocket,
    executeNotebookRerunAllWebSocket,
    cancelCellWebSocket,
    updateSourceWebSocket,
    setVariantActive,
    updateWidgetValues,
    setViewerMode,
    selectVariantDisplay,
    variantDisplayCellId,
    addVariant,
    // v1.1: Impact Preview, Profiling
    currentImpactPreview,
    profilingSummary,
    requestImpactPreview,
    clearImpactPreview,
    requestProfilingSummary,
    // Inspect REPL
    inspectCellId,
    inspectReady,
    inspectHistory,
    inspectOpenOrder,
    isInspecting,
    inspectReadyFor,
    inspectHistoryFor,
    maxInspectPanels,
    setMaxInspectPanels,
    pendingInspectRequest,
    confirmInspectSwap,
    cancelInspectSwap,
    openInspect,
    evalInspect,
    closeInspect,
    // Cell unit tests
    isTesting,
    toggleTests,
    closeTests,
    updateTestSource,
    runCellTests,
    // Environment / Dependencies
    dependencies,
    dependencyLoading,
    dependencyError,
    resolvedDependencies,
    environmentLoading,
    environmentError,
    environmentWarnings,
    environmentLastAction,
    environmentOperation,
    environmentJobHistory,
    environmentMutationActive,
    environmentImportPreview,
    availableWorkers,
    workerDefinitionsEditable,
    registryEnabled,
    workerHealthLoading,
    workerHealthCheckedAt,
    notebookWorkerError,
    workerRegistryError,
    serverManagedWorkers,
    serverWorkerRegistryAvailable,
    serverWorkerRegistryLoading,
    isServerWorkerActionLoading,
    serverWorkerRegistryError,
    fetchWorkers,
    ensureWorkersLoaded,
    fetchServerWorkerRegistry,
    fetchDependencies,
    fetchEnvironment,
    fetchRPackagesAction,
    addDependencyAction,
    initRenvAction,
    addRPackageAction,
    updatePythonVersionAction,
    removeDependencyAction,
    syncEnvironmentAction,
    exportRequirementsAction,
    previewRequirementsImportAction,
    importRequirementsAction,
    previewEnvironmentYamlImportAction,
    importEnvironmentYamlAction,
    clearEnvironmentImportPreview,
    updateNotebookWorkersAction,
    updateServerWorkerRegistryAction,
    saveServerWorkerAction,
    deleteServerWorkerAction,
    updateServerWorkerEnabledAction,
    refreshServerWorkerAction,
    updateNotebookWorkerAction,
    updateNotebookTimeoutAction,
    updateNotebookEnvAction,
    refreshSecretManagerAction,
    updateNotebookSecretManagerConfigAction,
    updateNotebookMountsAction,
    updateNotebookConnectionsAction,
    getConnectionSchemaAction,
    // LLM assistant
    llmAvailable,
    llmModel,
    llmProvider,
    llmLoading,
    llmError,
    llmMessages,
    checkLlmStatus,
    llmChat,
    clearLlmHistory,
    insertLlmCodeAsCell,
    insertLlmCodeAsCells,
    // Agent
    agentRunning,
    agentProgress,
    agentError,
    agentApprovalPrompts,
    agentAutoApprove,
    runAgentAction,
    cancelAgent,
    respondToApproval,
    resetAgentMemory,
  }
}
