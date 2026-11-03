/**
 * The one badge on a cell's run state. A stale result is no longer current, so a stale
 * cell says that rather than how its last run got its result.
 */
export function cellRunBadge(
  status: string,
  cacheHit: boolean | undefined,
): 'stale' | 'cached' | null {
  if (status === 'stale') return 'stale'
  return cacheHit ? 'cached' : null
}

/** The status dot's tooltip: its status, then the details the cell header leaves out. */
export function cellDotTitle(
  statusTitle: string,
  details: { isLeaf: boolean; executionMethod: string; hasRunBadge: boolean },
): string {
  const lines = [statusTitle]
  if (details.isLeaf) lines.push('Leaf: no downstream consumers')
  if (details.executionMethod && !details.hasRunBadge) {
    lines.push(`Executed via ${details.executionMethod} process`)
  }
  return lines.join('\n')
}
