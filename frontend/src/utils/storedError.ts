import type { CellOutput } from '../types/notebook'

/**
 * The failure a cell comes back with after a reload, which no `cell_error` frame replays. The
 * server sends the error only while it describes the cell's source. A cell back to running or
 * stale is not claiming that failure, so its output stays as it is.
 */
export function storedErrorOutput(cell: {
  status?: unknown
  error?: unknown
  error_full?: unknown
}): CellOutput | undefined {
  if (cell.status !== 'error' || typeof cell.error !== 'string' || !cell.error) return undefined
  return {
    contentType: 'json/object',
    error: cell.error,
    traceback: cell.error,
    tracebackFull: typeof cell.error_full === 'string' ? cell.error_full : undefined,
  }
}
