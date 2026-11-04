/**
 * Which columns hold numbers, so a column lines up as one: its header, its stats and
 * every cell, a null among them included. A column needs a number and nothing else but
 * nulls and the names the server gives what JSON cannot carry, so an all-null or mixed
 * column stays with the text.
 */
const NON_FINITE = new Set<unknown>(['NaN', 'inf', '-inf'])

export function numericColumns(rows: unknown[][], columnCount: number): boolean[] {
  return Array.from({ length: columnCount }, (_, column) => {
    const present = rows.map((row) => row[column]).filter((v) => v !== null && v !== undefined)
    return (
      present.some((v) => typeof v === 'number') &&
      present.every((v) => typeof v === 'number' || NON_FINITE.has(v))
    )
  })
}
