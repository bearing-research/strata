/**
 * The one badge on a cell's run state. A stale result is no longer current, so a stale
 * cell says that rather than how its last run got its result.
 */
export function cellRunBadge(
  status: string,
  cacheHit: boolean | undefined,
  executionMethod: string,
): 'stale' | 'cached' | 'method' | null {
  if (status === 'stale') return 'stale'
  if (cacheHit) return 'cached'
  return executionMethod ? 'method' : null
}
