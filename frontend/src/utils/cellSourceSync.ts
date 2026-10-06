/**
 * Whether a backend snapshot's cell source may replace the local one.
 *
 * Edits are local-first, but a cell can also change outside this tab (an agent
 * over the CLI or MCP) and the editor must follow. Rule: unflushed keystrokes
 * win, everything else yields to the backend.
 *
 * Don't also hold off while a flush is in flight: holding drops the update
 * rather than deferring it, so with no later snapshot the cell stops following
 * remote edits. The race is benign; the next snapshot brings the text back.
 */

export function shouldAdoptRemoteSource(params: {
  /** The snapshot's source. Anything but a string means the payload omitted it. */
  remote: unknown
  /** The buffer currently in the editor. */
  local: string
  /** Whether the cell has keystrokes not yet flushed to the backend. */
  isDirty: boolean
}): boolean {
  const { remote, local, isDirty } = params
  if (typeof remote !== 'string') return false
  if (isDirty) return false
  return local !== remote
}

/**
 * Cells whose editor text the backend has not taken yet.
 *
 * A cell stays dirty until its `cell_source_update` actually goes out, so text
 * typed while the socket is down is sent on reconnect and no snapshot replaces
 * it meanwhile. An edit refused with `cell_busy` (the cell was running) comes
 * back dirty and is resent once the cell's run settles.
 */
export class DirtySources {
  private readonly dirty = new Set<string>()
  private readonly heldForRun = new Set<string>()
  private readonly send: (cellId: string) => boolean

  /** `send` sends the cell's current text and says whether the frame went out. */
  constructor(send: (cellId: string) => boolean) {
    this.send = send
  }

  mark(cellId: string) {
    this.dirty.add(cellId)
  }

  has(cellId: string): boolean {
    return this.dirty.has(cellId)
  }

  flush(cellId: string) {
    if (this.dirty.has(cellId) && this.send(cellId)) this.dirty.delete(cellId)
  }

  flushAll() {
    for (const cellId of [...this.dirty]) this.flush(cellId)
  }

  refusedWhileRunning(cellId: string) {
    this.dirty.add(cellId)
    this.heldForRun.add(cellId)
  }

  /** A `cell_status` arrived: a held edit goes out once the cell stops running. */
  statusChanged(cellId: string, status: string) {
    if (status !== 'running' && this.heldForRun.delete(cellId)) this.flush(cellId)
  }
}
