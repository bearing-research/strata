// A widget cell's `# @live` line, read and edited where the server reads it: the
// leading comment block (`annotations.iter_annotation_block`), the last line winning,
// `off`/`false`/`no`/`0` turning it off (`vocabulary._flag`).

const LIVE = /^#\s*@live\b\s*(.*?)\s*$/
const OFF = new Set(['off', 'false', 'no', '0'])

/** The leading comment block's line count: up to the first line of code. */
function blockEnd(lines: string[]): number {
  const end = lines.findIndex((line) => line.trim() !== '' && !line.trim().startsWith('#'))
  return end < 0 ? lines.length : end
}

/** Whether *source* is live, as the server reads it. */
export function isLive(source: string): boolean {
  const lines = source.split('\n')
  const values = lines
    .slice(0, blockEnd(lines))
    .map((line) => LIVE.exec(line.trim()))
    .filter((match) => match !== null)
  const last = values.at(-1)
  return last !== undefined && !OFF.has(last[1].toLowerCase())
}

/** *source* with `# @live` on (first line) or off (no `@live` line in the block). */
export function withLive(source: string, live: boolean): string {
  const lines = source.split('\n')
  const end = blockEnd(lines)
  const kept = lines.filter((line, i) => i >= end || !LIVE.test(line.trim()))
  return (live ? ['# @live', ...kept] : kept).join('\n')
}
