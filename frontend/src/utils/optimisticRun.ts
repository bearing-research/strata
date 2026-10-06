// The Run buttons show a cell running before the server answers. A run refused
// because another holds the notebook (an `error` frame with code
// `notebook_busy` and the refused `cell_id`) gets no status frame after it, so
// the cell goes back to what it showed before the click.

export class OptimisticRuns<S extends string = string> {
  private readonly before = new Map<string, S>()

  /** The cell is about to show `running`; remember what it showed. */
  start(cellId: string, current: S) {
    // Already running: keep the status from before the first click.
    if (current !== 'running') this.before.set(cellId, current)
  }

  /** The server sent this cell's status, which is authoritative from here. */
  settle(cellId: string) {
    this.before.delete(cellId)
  }

  /** The status to restore when `payload` refuses one of these runs, else null. */
  refused(payload: unknown): { cellId: string; status: S } | null {
    if (!payload || typeof payload !== 'object') return null
    const { code, cell_id: cellId } = payload as { code?: unknown; cell_id?: unknown }
    if (code !== 'notebook_busy' || typeof cellId !== 'string') return null
    const status = this.before.get(cellId)
    if (status === undefined) return null
    this.before.delete(cellId)
    return { cellId, status }
  }
}
