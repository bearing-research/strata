// The names a cell reads from other cells: what its incoming DAG edges carry. A reference with
// no edge is the cell's own business, such as the ambient `strata` client or an undefined name.
export function cellReads(
  cellId: string,
  references: string[],
  edges: { to_cell_id: string; variable: string }[],
): string[] {
  const carried = new Set(edges.filter((e) => e.to_cell_id === cellId).map((e) => e.variable))
  return references.filter((name) => carried.has(name))
}
