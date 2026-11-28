// A cell's HTML output runs in a sandboxed frame; all it may tell the notebook is its height.

export const MAX_HTML_OUTPUT_HEIGHT = 4000

/** The height *event* reports, if it came from *frame* and is a usable number, else null. */
export function reportedHeight(
  event: Pick<MessageEvent, 'source' | 'data'>,
  frame: Window | null | undefined,
): number | null {
  if (!frame || event.source !== frame) return null
  const height = (event.data as { strataOutputHeight?: unknown } | null)?.strataOutputHeight
  if (typeof height !== 'number' || !Number.isFinite(height) || height <= 0) return null
  return Math.min(Math.ceil(height), MAX_HTML_OUTPUT_HEIGHT)
}

/**
 * The heights to give the frame, as a page reports them. A page sized to the viewport
 * (``100vh`` plus a margin) grows by the same step each time the frame does: on the
 * second such step in a row, the frame stops growing instead of chasing it to the cap.
 */
export function heightTracker(): (height: number) => number | null {
  let current: number | null = null
  let lastGrowth: number | null = null
  let chasing = false
  return (height) => {
    if (current !== null && height > current) {
      const growth = height - current
      if (chasing || (lastGrowth !== null && Math.abs(growth - lastGrowth) <= 1)) {
        chasing = true
        return null
      }
      lastGrowth = growth
    } else {
      lastGrowth = null
    }
    current = height
    return height
  }
}
