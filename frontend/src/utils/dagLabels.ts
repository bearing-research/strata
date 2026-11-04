// Edge labels in the DAG. Edges that converge share a midpoint, so a label moves off an
// occupied spot, and is dropped when no spot is free; its edge's tooltip still names it.

export interface Box {
  x: number
  y: number
  width: number
  height: number
}

export const LABEL_CHAR_WIDTH = 5.4 // the 9px monospace the labels are drawn in
export const LABEL_HEIGHT = 11

function overlaps(a: Box, b: Box): boolean {
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

/** Where a label starts: on the side its edge bends away from, so the line misses the text. */
export function labelAnchor(
  mid: { x: number; y: number },
  fromX: number,
  toX: number,
  text: string,
): { x: number; y: number; text: string } {
  const x = toX < fromX ? mid.x - 6 - text.length * LABEL_CHAR_WIDTH : mid.x + 6
  return { x, y: mid.y - 6, text }
}

/** Each label's top-left corner, or null when it has no free spot. */
export function placeEdgeLabels(
  labels: { x: number; y: number; text: string }[],
  nodes: Box[],
): ({ x: number; y: number } | null)[] {
  const placed: Box[] = []
  return labels.map(({ x, y, text }) => {
    const width = text.length * LABEL_CHAR_WIDTH
    const candidates = [
      [x, y],
      [x, y - LABEL_HEIGHT - 2],
      [x, y + LABEL_HEIGHT + 2],
      [x - width - 12, y],
    ]
    for (const [cx, cy] of candidates) {
      const box = { x: cx, y: cy, width, height: LABEL_HEIGHT }
      if (!placed.some((p) => overlaps(p, box)) && !nodes.some((n) => overlaps(n, box))) {
        placed.push(box)
        return { x: cx, y: cy }
      }
    }
    return null
  })
}
