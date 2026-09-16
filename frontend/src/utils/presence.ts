import type { PresenceEntryModel } from '../types/ws-payloads.generated'

/** The others on the session whose focus is *cellId*, leaving out *you*. */
export function othersOnCell(
  entries: PresenceEntryModel[],
  cellId: string,
  you: string | null,
): string[] {
  return entries
    .filter((entry) => entry.focused_cell_id === cellId && entry.principal !== you)
    .map((entry) => entry.principal)
}
