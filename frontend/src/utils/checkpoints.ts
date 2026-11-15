/** A cell's checkpoints as `GET /cells/{id}/checkpoints` lists them, and how the panel words them. */

export type CheckpointRunStatus = 'running' | 'interrupted' | 'finished' | 'failed'

export interface CheckpointInfo {
  /** What a branch from this checkpoint is started with. */
  releaseId: string
  sequence: number
  /** Unix seconds. */
  createdAt: number
  parts: string[]
  files: number
  step: number | null
  byteSize: number
  heads: string[]
}

export interface CheckpointRunInfo {
  buildId: string
  /** Null once the ledger no longer has the run's row. */
  status: CheckpointRunStatus | null
  variant: string | null
  /** The release this run branched from, if it is a branch. */
  init: string | null
  startedAt: number | null
  checkpoints: CheckpointInfo[]
}

type Raw = Record<string, unknown>

const STATUSES = new Set(['running', 'interrupted', 'finished', 'failed'])

function nullable<T>(value: unknown, convert: (value: unknown) => T): T | null {
  return value === null || value === undefined ? null : convert(value)
}

export function parseCheckpointRuns(runs: Raw[]): CheckpointRunInfo[] {
  return runs.map((run) => ({
    buildId: String(run.build_id),
    status: STATUSES.has(String(run.status)) ? (run.status as CheckpointRunStatus) : null,
    variant: nullable(run.variant, String),
    init: nullable(run.init, String),
    startedAt: nullable(run.started_at, Number),
    checkpoints: (run.checkpoints as Raw[]).map((checkpoint) => ({
      releaseId: String(checkpoint.release_id),
      sequence: Number(checkpoint.sequence),
      createdAt: Number(checkpoint.created_at),
      parts: (checkpoint.parts as unknown[]).map(String),
      files: Number(checkpoint.files),
      step: nullable(checkpoint.step, Number),
      byteSize: Number(checkpoint.byte_size),
      heads: (checkpoint.heads as unknown[]).map(String),
    })),
  }))
}

/** What a checkpoint holds: its parts by name, and its files by count. */
export function checkpointContents(checkpoint: CheckpointInfo): string {
  const held = [...checkpoint.parts]
  if (checkpoint.files > 0) {
    held.push(`${checkpoint.files} ${checkpoint.files === 1 ? 'file' : 'files'}`)
  }
  return held.join(', ')
}

/**
 * The checkpoint a branch started from, when it is still listed. Prune may have
 * collected it since: the branch stays, and its origin is then not shown.
 */
export function branchOrigin(
  run: CheckpointRunInfo,
  runs: CheckpointRunInfo[],
): { run: CheckpointRunInfo; checkpoint: CheckpointInfo } | null {
  if (run.init === null) return null
  for (const other of runs) {
    const checkpoint = other.checkpoints.find((c) => c.releaseId === run.init)
    if (checkpoint) return { run: other, checkpoint }
  }
  return null
}
