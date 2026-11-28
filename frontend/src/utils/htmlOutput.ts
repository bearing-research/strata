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
