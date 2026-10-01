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
