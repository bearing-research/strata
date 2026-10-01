/**
 * How a cell's authorship reads in its header.
 *
 * The browser records `local`, so badging it would mark every cell and hide
 * the agent-written ones. Only other authors get a badge.
 */

export const LOCAL_AUTHOR = 'local'

export function authorBadgeLabel(
  createdBy: string | null | undefined,
  updatedBy: string | null | undefined,
): string | null {
  if (updatedBy && updatedBy !== LOCAL_AUTHOR) return `by ${updatedBy}`
  // Agent-created but edited here: the origin is still what matters.
  if (createdBy && createdBy !== LOCAL_AUTHOR) return `added by ${createdBy}`
  return null
}

export function authorTitle(
  createdBy: string | null | undefined,
  updatedBy: string | null | undefined,
): string {
  return `Added by ${createdBy || 'unrecorded'} · last edited by ${updatedBy || 'unrecorded'}`
}
