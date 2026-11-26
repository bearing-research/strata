// A widget cell's `# @live` line, edited where the server reads it: the leading
// comment block (`annotations.iter_annotation_block`), not anywhere in the source.

const LIVE = /^#\s*@live\b/

/** *source* with `# @live` on (first line) or off (no `@live` line in the block). */
export function withLive(source: string, live: boolean): string {
  const lines = source.split('\n')
  let end = lines.findIndex((line) => line.trim() !== '' && !line.trim().startsWith('#'))
  if (end < 0) end = lines.length
  const kept = lines.filter((line, i) => i >= end || !LIVE.test(line.trim()))
  return (live ? ['# @live', ...kept] : kept).join('\n')
}
