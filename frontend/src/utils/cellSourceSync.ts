/**
 * Deciding whether a backend snapshot's cell source may replace the local one.
 *
 * Source edits are local-first: typing updates the buffer and marks the cell
 * dirty, and the flush happens later. But a cell can also change from outside
 * this tab — an agent driving the notebook over the CLI or MCP, or
 * `strata cell edit` — and those edits have to reach the editor, or a human
 * watching an agent sees cells go stale above source that still looks
 * unchanged.
 *
 * One guard resolves the tension: unflushed keystrokes win, everything else
 * yields to the backend.
 *
 * A first attempt also held off while a flush was sent but not yet echoed
 * back, to avoid adopting a snapshot built before the backend applied it.
 * That was worse than the race it prevented. Holding *drops* the update rather
 * than deferring it — the decision is only revisited when another snapshot
 * arrives, and if none does, the cell stops following remote edits for the
 * life of the page. The race itself is benign: the flush is already on its way,
 * so the backend's next snapshot carries the text back and the worst case is a
 * brief flicker. And if the flush never lands, the file on disk really does
 * hold the old text, so showing it is honest.
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
