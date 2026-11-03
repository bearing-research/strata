/** A duration as the cell header and the profiling panel write it. */
export function formatDuration(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`
}

/**
 * A cache hit's badge text: what it saved when that is known, never "saved 0ms".
 * Plain "cached" too when the load took longer than the run did: nothing was saved.
 */
export function cacheBadgeLabel(
  savedMs: number | null | undefined,
  loadMs: number | null | undefined,
): string {
  if (!savedMs || savedMs < (loadMs ?? 0)) return 'cached'
  return `cached · saved ${formatDuration(savedMs)}`
}

/** A cache hit's tooltip: the load time, which the badge no longer shows. */
export function cacheBadgeTitle(loadMs: number | null | undefined): string {
  return loadMs ? `Result loaded from cache in ${loadMs}ms` : 'Result loaded from cache'
}
