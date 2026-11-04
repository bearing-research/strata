/** Notebooks in the storage directory that the recents list above them doesn't already show. */
export function storedNotInRecents<T extends { path: string }>(
  stored: T[],
  recents: { path: string }[],
): T[] {
  const shown = new Set(recents.map((entry) => entry.path))
  return stored.filter((notebook) => !shown.has(notebook.path))
}
