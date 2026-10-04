/**
 * The notebook cells that read a registry name with `# @dataset`, as the
 * registry dashboard lists them.
 */

export interface DatasetReader {
  notebook_id: string
  cell_id: string
  /** The name as the cell declared it: `name`, `name@alias` or `name@v=<n>`. */
  reference: string
}

/** This notebook's readers, then the ones the registry recorded, each cell once. */
export function mergeReaders(
  fromRegistry: DatasetReader[],
  local: DatasetReader[],
): DatasetReader[] {
  const seen = new Set<string>()
  const merged: DatasetReader[] = []
  for (const reader of [...local, ...fromRegistry]) {
    const key = `${reader.notebook_id}\u0000${reader.cell_id}`
    if (seen.has(key)) continue
    seen.add(key)
    merged.push(reader)
  }
  return merged
}

/** A cell of this notebook by its name; any other with the notebook it lives in. */
export function readerLabel(
  reader: DatasetReader,
  notebookId: string,
  cellNames: Record<string, string>,
): string {
  if (reader.notebook_id === notebookId) return cellNames[reader.cell_id] || reader.cell_id
  return `${reader.notebook_id.slice(0, 8)}/${reader.cell_id}`
}
