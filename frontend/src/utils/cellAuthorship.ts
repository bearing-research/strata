/**
 * How a cell's authorship reads in its header.
 *
 * `local` is what the browser records, so on a notebook you write yourself it
 * is every cell. Badging it would put the same word on every cell and make the
 * agent-written ones — the ones worth noticing — no easier to find. So a badge
 * appears only when someone other than the person at this browser wrote or
 * last edited the cell.
 */

export const LOCAL_AUTHOR = 'local'

export function authorBadgeLabel(
  createdBy: string | null | undefined,
  updatedBy: string | null | undefined,
): string | null {
  if (updatedBy && updatedBy !== LOCAL_AUTHOR) return `by ${updatedBy}`
  // Written by an agent and since edited here: the origin is still worth
  // saying, since "which cells did the agent start" is the question.
  if (createdBy && createdBy !== LOCAL_AUTHOR) return `added by ${createdBy}`
  return null
}

export function authorTitle(
  createdBy: string | null | undefined,
  updatedBy: string | null | undefined,
): string {
  return `Added by ${createdBy || 'unrecorded'} · last edited by ${updatedBy || 'unrecorded'}`
}
