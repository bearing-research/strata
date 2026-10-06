// The console contract (docs/reference/websocket.md): `cell_status: running`
// clears it, `cell_console` appends per stream, and `cell_output` /
// `cell_error` replace it with their `stdout` / `stderr` strings.

export interface CellConsole {
  consoleStdout?: string
  consoleStderr?: string
}

// A cell's console after one `cell_console` frame. Chunk 0 of a remote run's
// stream starts it afresh, so the last run's text does not stay above the new
// run's; every other frame appends.
export function applyConsoleChunk(
  existing: string | undefined,
  text: string,
  chunkSeq: unknown,
): string {
  return (chunkSeq === 0 ? '' : (existing ?? '')) + text
}

// `cell_console`: one frame of `text` on its `stream`.
export function appendConsole(cell: CellConsole, payload: Record<string, unknown>): void {
  const text = typeof payload.text === 'string' ? payload.text : ''
  if (!text) return
  if (payload.stream === 'stderr') {
    cell.consoleStderr = applyConsoleChunk(cell.consoleStderr, text, payload.chunk_seq)
  } else {
    cell.consoleStdout = applyConsoleChunk(cell.consoleStdout, text, payload.chunk_seq)
  }
}

// `cell_status: running`: the run starts with an empty console.
export function startConsoleRun(cell: CellConsole): void {
  cell.consoleStdout = ''
  cell.consoleStderr = ''
}

// `cell_output` / `cell_error`: their `stdout` / `stderr` are the run's whole
// console. An absent stream keeps what was shown.
export function replaceConsole(cell: CellConsole, payload: Record<string, unknown>): void {
  if (typeof payload.stdout === 'string') cell.consoleStdout = payload.stdout
  if (typeof payload.stderr === 'string') cell.consoleStderr = payload.stderr
}
