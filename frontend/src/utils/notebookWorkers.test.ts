import assert from 'node:assert/strict'
import test from 'node:test'

import type { WorkerCatalogEntry } from '../types/notebook.ts'
import {
  applyWorkerHealth,
  workerAttentionReason,
  workerNeedsAttention,
  summarizeRemoteExecutionState,
  summarizeWorkerCatalog,
  editorWorkerTransport,
  effectiveWorkerNameForCell,
  isRemoteExecutorLikelyUnreachable,
  resolveEffectiveWorkerEntry,
  summarizeRemoteExecutionIssue,
  workerTransportLabel,
  workerWarningForEntry,
} from './notebookWorkers.ts'

function makeWorker(overrides: Partial<WorkerCatalogEntry> = {}): WorkerCatalogEntry {
  return {
    name: 'gpu-http',
    backend: 'executor',
    runtimeId: 'gpu-a100',
    config: { url: 'https://executor.example/v1/execute' },
    source: 'server',
    health: 'healthy',
    allowed: true,
    ...overrides,
  }
}

test('workerTransportLabel classifies local, embedded, direct, and signed workers', () => {
  assert.equal(workerTransportLabel(makeWorker({ backend: 'local', config: {} })), 'local')
  assert.equal(
    workerTransportLabel(makeWorker({ config: { url: 'embedded://local' } })),
    'embedded',
  )
  assert.equal(
    workerTransportLabel(makeWorker({ config: { url: 'https://executor.example/v1/execute' } })),
    'direct',
  )
  assert.equal(
    workerTransportLabel(
      makeWorker({
        config: {
          url: 'https://executor.example/v1/execute',
          transport: 'signed',
        },
      }),
    ),
    'signed',
  )
})

test('editorWorkerTransport keeps the signed aliases signed', () => {
  for (const transport of ['signed', 'manifest', 'build', ' Manifest ']) {
    assert.equal(editorWorkerTransport(transport), 'signed', transport)
  }
  for (const transport of ['direct', undefined, null, '']) {
    assert.equal(editorWorkerTransport(transport), 'direct', String(transport))
  }
})

test('resolveEffectiveWorkerEntry returns synthetic local and configured remote workers', () => {
  const workers = [makeWorker()]

  assert.equal(resolveEffectiveWorkerEntry(workers, 'local')?.name, 'local')
  assert.equal(resolveEffectiveWorkerEntry(workers, 'gpu-http')?.name, 'gpu-http')
  assert.equal(resolveEffectiveWorkerEntry(workers, 'missing'), null)
})

test('effectiveWorkerNameForCell prefers source annotations over persisted worker config', () => {
  assert.equal(
    effectiveWorkerNameForCell({
      worker: 'gpu-default',
      annotations: { worker: 'gpu-override', timeout: null, env: {}, mounts: [] },
    }),
    'gpu-override',
  )
  assert.equal(
    effectiveWorkerNameForCell({
      worker: 'gpu-default',
    }),
    'gpu-default',
  )
  assert.equal(
    effectiveWorkerNameForCell({
      worker: null,
    }),
    'local',
  )
})

test('workerWarningForEntry reports unresolved, blocked, and unreachable workers', () => {
  assert.equal(workerWarningForEntry(null, 'gpu-missing'), 'Worker "gpu-missing" is unresolved')
  assert.equal(
    workerWarningForEntry(makeWorker({ enabled: false }), 'gpu-http'),
    'Worker "gpu-http" is disabled by server policy',
  )
  assert.equal(
    workerWarningForEntry(makeWorker({ allowed: false }), 'gpu-http'),
    'Worker "gpu-http" is blocked by server policy',
  )
  assert.equal(
    workerWarningForEntry(makeWorker({ health: 'unavailable' }), 'gpu-http'),
    'Worker "gpu-http" is currently unreachable',
  )
  assert.equal(workerWarningForEntry(makeWorker(), 'gpu-http'), null)
})

test('worker attention helpers classify unhealthy or policy-blocked workers', () => {
  assert.equal(workerNeedsAttention(makeWorker()), false)
  assert.equal(workerNeedsAttention(makeWorker({ enabled: false })), true)
  assert.equal(workerNeedsAttention(makeWorker({ health: 'unavailable' })), true)
  assert.equal(workerNeedsAttention(makeWorker({ consecutiveFailures: 2 })), true)
  assert.equal(workerAttentionReason(makeWorker({ allowed: false })), 'Blocked by server policy')
  assert.equal(
    workerAttentionReason(makeWorker({ health: 'unavailable' })),
    'Worker is currently unreachable',
  )
  assert.equal(
    workerAttentionReason(makeWorker({ consecutiveFailures: 2 })),
    '2 consecutive failures',
  )
})

test('summarizeWorkerCatalog reports worker totals and attention counts', () => {
  const summary = summarizeWorkerCatalog([
    makeWorker({ name: 'healthy', health: 'healthy' }),
    makeWorker({ name: 'down', health: 'unavailable' }),
    makeWorker({ name: 'disabled', enabled: false, health: 'unknown' }),
  ])

  assert.deepEqual(summary, {
    total: 3,
    healthy: 1,
    unavailable: 1,
    unknown: 1,
    disabled: 1,
    blocked: 0,
    attention: 2,
  })
})

test('isRemoteExecutorLikelyUnreachable recognizes transport/connectivity failures', () => {
  assert.equal(
    isRemoteExecutorLikelyUnreachable(
      'Execution failed: Remote executor request failed for worker "gpu-http": All connection attempts failed',
    ),
    true,
  )
  assert.equal(
    isRemoteExecutorLikelyUnreachable(
      'Execution failed: Remote executor request failed for worker "gpu-http": Connection refused',
    ),
    true,
  )
  assert.equal(
    isRemoteExecutorLikelyUnreachable(
      'Execution failed: Remote executor "gpu-http" returned 500: harness failed',
    ),
    false,
  )
})

test('summarizeRemoteExecutionIssue generates user-facing remote execution summaries', () => {
  const worker = makeWorker()

  assert.equal(
    summarizeRemoteExecutionIssue(
      'Execution failed: Remote executor request failed for worker "gpu-http": All connection attempts failed',
      worker,
      worker.name,
      'REQUEST_FAILED',
    ),
    'Could not reach remote worker "gpu-http"',
  )
  assert.equal(
    summarizeRemoteExecutionIssue(
      'Execution failed: Failed to finalize notebook bundle build',
      worker,
      worker.name,
      'FINALIZE_FAILED',
    ),
    'Remote execution finished, but output upload/finalize failed',
  )
  assert.equal(
    summarizeRemoteExecutionIssue(
      'Execution failed: worker is not allowed in service mode',
      worker,
      worker.name,
    ),
    'Worker "gpu-http" is blocked by policy',
  )
  assert.equal(
    summarizeRemoteExecutionIssue(
      'Execution failed: Notebook build failed',
      worker,
      worker.name,
      null,
      'failed',
    ),
    'Remote build failed on "gpu-http"',
  )
})

test('summarizeRemoteExecutionState describes successful, cached, and failed remote runs', () => {
  assert.deepEqual(
    summarizeRemoteExecutionState({
      executionMethod: 'executor',
      remoteWorkerName: 'gpu-http',
      remoteTransport: 'direct',
      hasError: false,
    }),
    {
      label: 'Remote run',
      detail: 'Executed via direct transport',
      tone: 'info',
    },
  )
  assert.deepEqual(
    summarizeRemoteExecutionState({
      executionMethod: 'cached',
      remoteWorkerName: 'gpu-http',
      remoteTransport: 'signed',
      remoteBuildState: 'ready',
      hasError: false,
    }),
    {
      label: 'Remote cache',
      detail: 'Loaded from cache for a previous remote execution',
      tone: 'success',
    },
  )
  assert.deepEqual(
    summarizeRemoteExecutionState({
      executionMethod: 'executor',
      remoteWorkerName: 'gpu-http',
      remoteTransport: 'signed',
      remoteBuildState: 'failed',
      remoteErrorCode: 'FINALIZE_FAILED',
      hasError: true,
    }),
    {
      label: 'Remote failure',
      detail: 'Remote execution finished, but output upload or finalize failed',
      tone: 'error',
    },
  )
})

test('applyWorkerHealth updates only the targeted remote worker entry', () => {
  const before = [
    makeWorker({ name: 'gpu-http', health: 'healthy' }),
    makeWorker({ name: 'gpu-signed', health: 'unknown' }),
  ]

  const after = applyWorkerHealth(before, 'gpu-signed', 'unavailable')

  assert.equal(after[0].health, 'healthy')
  assert.equal(after[1].health, 'unavailable')
  assert.notEqual(after[1], before[1])
})
