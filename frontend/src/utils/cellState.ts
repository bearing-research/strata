// The runtime states a cell can show, as frontend/DESIGN.md defines them.
import { Ban, Circle, CircleDashed, Clock, Contrast, Send, X, Zap } from '@lucide/vue'

export const CELL_STATES = {
  idle: { label: 'idle', icon: Circle },
  queued: { label: 'queued', icon: Clock },
  running: { label: 'running', icon: CircleDashed },
  dispatching: { label: 'dispatching', icon: Send },
  ready: { label: 'ready', icon: Circle },
  cached: { label: 'cached', icon: Zap },
  stale: { label: 'stale', icon: Contrast },
  error: { label: 'error', icon: X },
  blocked: { label: 'blocked', icon: Ban },
}

export type CellState = keyof typeof CELL_STATES

/**
 * What a cell shows. A run in flight says so even if the source has since gained an annotation
 * error; otherwise that error wins, since it is what stops the next run.
 */
export function cellState(cell: {
  status: string
  output?: { cacheHit?: boolean }
  annotationDiagnostics?: { severity: string }[]
  remoteWorkerName?: string
  language?: string
}): CellState {
  // It never runs, so the status the server keeps for it says nothing to a reader.
  if (cell.language === 'markdown') return 'idle'
  if (cell.status === 'running') return cell.remoteWorkerName ? 'dispatching' : 'running'
  if (cell.status === 'queued') return 'queued'
  if (cell.annotationDiagnostics?.some((d) => d.severity === 'error')) return 'blocked'
  if (cell.status === 'error') return 'error'
  if (cell.status === 'stale') return 'stale'
  if (cell.status === 'ready') return cell.output?.cacheHit ? 'cached' : 'ready'
  return 'idle'
}
