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

/**
 * A stale cell back to ready without a run: an edit was undone, so its stored result is current
 * again, and it reads as a hit. A run goes through `running`, so it never lands here.
 */
export function isCachedAgain(previous: string | undefined, next: string): boolean {
  return previous === 'stale' && next === 'ready'
}

/** The fields of a result that is current again; one that was a hit keeps what it saved. */
export function cachedAgainFields(
  output: { cacheHit?: boolean; cacheSavedMs?: number },
  lastRunMs: number | undefined,
): { cacheHit: true; cacheSavedMs: number | undefined } {
  return { cacheHit: true, cacheSavedMs: output.cacheHit ? output.cacheSavedMs : lastRunMs }
}

/**
 * How a result was got, for the output a sync from the server has just rebuilt from its
 * display. A stale one keeps its saving: if the change goes away it is a hit again.
 */
export function cacheFieldsAfterSync(
  serverCacheHit: boolean | undefined,
  status: string,
  previous: { cacheHit?: boolean; cacheSavedMs?: number; cacheLoadMs?: number } | undefined,
  lastRunMs: number | undefined,
): { cacheHit?: true; cacheSavedMs?: number; cacheLoadMs?: number } {
  const cacheLoadMs = previous?.cacheLoadMs
  if (serverCacheHit) return { ...cachedAgainFields(previous ?? {}, lastRunMs), cacheLoadMs }
  if (status === 'stale' && previous?.cacheHit) {
    return { cacheHit: true, cacheSavedMs: previous.cacheSavedMs, cacheLoadMs }
  }
  return {}
}
