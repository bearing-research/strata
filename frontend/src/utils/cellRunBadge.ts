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
