import type { CellAnnotations, WorkerCatalogEntry, WorkerHealth } from '../types/notebook'

const LOCAL_WORKER_ENTRY: WorkerCatalogEntry = {
  name: 'local',
  backend: 'local',
  runtimeId: null,
  config: {},
  source: 'builtin',
  health: 'healthy',
  allowed: true,
  enabled: true,
  transport: 'local',
}

export function effectiveWorkerNameForCell(cell: {
  annotations?: CellAnnotations
  worker: string | null
}): string {
  return cell.annotations?.worker || cell.worker || 'local'
}

export function resolveEffectiveWorkerEntry(
  availableWorkers: WorkerCatalogEntry[],
  workerName: string | null | undefined,
): WorkerCatalogEntry | null {
  const effectiveName = workerName || 'local'
  if (effectiveName === 'local') {
    return { ...LOCAL_WORKER_ENTRY }
  }
  return availableWorkers.find((worker) => worker.name === effectiveName) ?? null
}

// Mirrors SIGNED_TRANSPORTS in workers.py: aliases of the signed-URL pull path.
const SIGNED_TRANSPORTS = new Set(['signed', 'manifest', 'build'])

export function editorWorkerTransport(transport: unknown): 'direct' | 'signed' {
  return SIGNED_TRANSPORTS.has(
    String(transport || 'direct')
      .trim()
      .toLowerCase(),
  )
    ? 'signed'
    : 'direct'
}

export function workerTransportLabel(
  worker: Pick<WorkerCatalogEntry, 'backend' | 'config' | 'transport'>,
): string {
  if (worker.transport) return worker.transport
  if (worker.backend === 'local') return 'local'
  const url = String(worker.config?.url || '')
  const transport = String(worker.config?.transport || 'direct')
    .trim()
    .toLowerCase()
  if (url.startsWith('embedded://')) return 'embedded'
  if (SIGNED_TRANSPORTS.has(transport)) return 'signed'
  if (url.startsWith('http://') || url.startsWith('https://')) return 'direct'
  return 'executor'
}

export interface WorkerCatalogSummary {
  total: number
  healthy: number
  unavailable: number
  unknown: number
  disabled: number
  blocked: number
  attention: number
}

export interface RemoteExecutionStateSummary {
  label: string
  detail: string
  tone: 'info' | 'success' | 'warning' | 'error'
}

export function workerNeedsAttention(worker: WorkerCatalogEntry): boolean {
  return Boolean(
    worker.enabled === false ||
    worker.allowed === false ||
    worker.health === 'unavailable' ||
    (worker.consecutiveFailures ?? 0) > 0 ||
    worker.lastError,
  )
}

export function workerAttentionReason(worker: WorkerCatalogEntry): string | null {
  if (worker.enabled === false) return 'Disabled by server policy'
  if (worker.allowed === false) return 'Blocked by server policy'
  if (worker.health === 'unavailable') return 'Worker is currently unreachable'
  if ((worker.consecutiveFailures ?? 0) > 0) {
    const failures = worker.consecutiveFailures ?? 0
    return failures === 1 ? '1 consecutive failure' : `${failures} consecutive failures`
  }
  if (worker.lastError) return worker.lastError
  return null
}

export function summarizeWorkerCatalog(workers: WorkerCatalogEntry[]): WorkerCatalogSummary {
  return workers.reduce<WorkerCatalogSummary>(
    (summary, worker) => {
      summary.total += 1

      if (worker.health === 'healthy') summary.healthy += 1
      else if (worker.health === 'unavailable') summary.unavailable += 1
      else summary.unknown += 1

      if (worker.enabled === false) summary.disabled += 1
      if (worker.allowed === false) summary.blocked += 1
      if (workerNeedsAttention(worker)) summary.attention += 1

      return summary
    },
    {
      total: 0,
      healthy: 0,
      unavailable: 0,
      unknown: 0,
      disabled: 0,
      blocked: 0,
      attention: 0,
    },
  )
}

export function workerWarningForEntry(
  entry: WorkerCatalogEntry | null,
  workerLabel: string,
): string | null {
  if (!entry) return `Worker "${workerLabel}" is unresolved`
  if (entry.enabled === false) return `Worker "${entry.name}" is disabled by server policy`
  if (entry.allowed === false) return `Worker "${entry.name}" is blocked by server policy`
  if (entry.backend === 'executor' && entry.health === 'unavailable') {
    return `Worker "${entry.name}" is currently unreachable`
  }
  return null
}

export function isRemoteExecutorLikelyUnreachable(error: string): boolean {
  const normalized = error.toLowerCase()
  return (
    normalized.includes('remote executor request failed') ||
    normalized.includes('connection refused') ||
    normalized.includes('all connection attempts failed') ||
    normalized.includes('name or service not known') ||
    normalized.includes('temporary failure in name resolution') ||
    normalized.includes('network is unreachable') ||
    normalized.includes('could not connect')
  )
}

export function summarizeRemoteExecutionIssue(
  error: string,
  entry: WorkerCatalogEntry | null,
  workerLabel: string,
  remoteErrorCode?: string | null,
  remoteBuildState?: string | null,
): string | null {
  if (!error || !entry || entry.backend !== 'executor') return null

  if (remoteErrorCode === 'TIMEOUT') {
    return `Remote execution timed out on "${entry.name}"`
  }
  if (remoteErrorCode === 'FINALIZE_FAILED') {
    return 'Remote execution finished, but output upload/finalize failed'
  }
  if (remoteErrorCode === 'INVALID_NOTEBOOK_BUNDLE') {
    return 'Remote execution returned an invalid output bundle'
  }
  if (remoteErrorCode === 'REQUEST_FAILED') {
    return `Could not reach remote worker "${entry.name}"`
  }
  if (remoteBuildState === 'failed') {
    return `Remote build failed on "${entry.name}"`
  }
  if (error.includes('blocked by server policy') || error.includes('not allowed in service mode')) {
    return `Worker "${entry.name}" is blocked by policy`
  }
  if (error.includes('Remote executor request failed')) {
    return `Could not reach remote worker "${entry.name}"`
  }
  if (error.includes('timed out')) {
    return `Remote execution timed out on "${entry.name}"`
  }
  if (error.includes('Failed to finalize notebook bundle build')) {
    return 'Remote execution finished, but output upload/finalize failed'
  }
  if (error.includes('Remote executor')) {
    return `Remote worker "${entry.name}" returned an execution error`
  }
  if (error.includes('Notebook build')) {
    return 'Remote execution did not complete successfully'
  }
  return `Remote execution failed on "${workerLabel}"`
}

export function summarizeRemoteExecutionState(params: {
  executionMethod?: string | null
  remoteWorkerName?: string | null
  remoteTransport?: string | null
  remoteBuildState?: string | null
  remoteErrorCode?: string | null
  hasError?: boolean
}): RemoteExecutionStateSummary | null {
  const {
    executionMethod,
    remoteWorkerName,
    remoteTransport,
    remoteBuildState,
    remoteErrorCode,
    hasError,
  } = params

  if (
    !remoteWorkerName &&
    !remoteTransport &&
    !remoteBuildState &&
    !remoteErrorCode &&
    executionMethod !== 'executor'
  ) {
    return null
  }

  if (hasError || remoteErrorCode || remoteBuildState === 'failed') {
    if (remoteErrorCode === 'TIMEOUT') {
      return {
        label: 'Remote failure',
        detail: 'Remote execution timed out before it could return results',
        tone: 'error',
      }
    }
    if (remoteErrorCode === 'FINALIZE_FAILED') {
      return {
        label: 'Remote failure',
        detail: 'Remote execution finished, but output upload or finalize failed',
        tone: 'error',
      }
    }
    if (remoteErrorCode === 'REQUEST_FAILED') {
      return {
        label: 'Remote failure',
        detail: 'The notebook could not reach the selected remote worker',
        tone: 'error',
      }
    }
    if (remoteBuildState === 'failed') {
      return {
        label: 'Remote failure',
        detail: 'The signed remote build failed before outputs were finalized',
        tone: 'error',
      }
    }
    return {
      label: 'Remote failure',
      detail: 'Remote execution did not complete successfully',
      tone: 'error',
    }
  }

  if (executionMethod === 'cached') {
    return {
      label: 'Remote cache',
      detail: 'Loaded from cache for a previous remote execution',
      tone: 'success',
    }
  }

  if (remoteTransport === 'signed') {
    return {
      label: 'Signed remote run',
      detail:
        remoteBuildState === 'ready'
          ? 'Remote execution and bundle finalize completed'
          : `Signed build state: ${remoteBuildState || 'unknown'}`,
      tone: remoteBuildState === 'ready' ? 'success' : 'warning',
    }
  }

  if (remoteTransport) {
    return {
      label: 'Remote run',
      detail: `Executed via ${remoteTransport} transport`,
      tone: 'info',
    }
  }

  if (remoteWorkerName || executionMethod === 'executor') {
    return {
      label: 'Remote run',
      detail: 'Executed on a remote worker',
      tone: 'info',
    }
  }

  return null
}

export function applyWorkerHealth(
  availableWorkers: WorkerCatalogEntry[],
  workerName: string,
  health: WorkerHealth,
): WorkerCatalogEntry[] {
  if (!workerName || workerName === 'local') return availableWorkers
  return availableWorkers.map((worker) =>
    worker.name === workerName ? { ...worker, health } : worker,
  )
}
