import type { Cell } from '../types/notebook.ts'

/**
 * The cells the app view shows: widgets, markdown, and cells with a display output,
 * unless marked `@app hide` (the backend reads it, as export does).
 */
export function appCells(cells: Cell[]): Cell[] {
  return cells.filter(
    (c) =>
      !c.annotations?.appHidden &&
      (c.language === 'widget' ||
        c.language === 'markdown' ||
        (c.displayOutputs !== undefined && c.displayOutputs.length > 0)),
  )
}
