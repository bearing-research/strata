/**
 * Which columns hold numbers, so a column lines up as one: its header, its stats and
 * every cell, a null among them included. A column needs a number and nothing else but
 * nulls, so an all-null or mixed column stays with the text.
 */
export function numericColumns(rows: unknown[][], columnCount: number): boolean[] {
  return Array.from({ length: columnCount }, (_, column) => {
    const present = rows.map((row) => row[column]).filter((v) => v !== null && v !== undefined)
    return present.length > 0 && present.every((v) => typeof v === 'number')
  })
}
