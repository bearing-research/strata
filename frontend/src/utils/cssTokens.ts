/**
 * Collect the CSS custom properties the frontend defines and references.
 *
 * A missing token fails silently (the fallback literal is used, or the
 * declaration is dropped) and no build, type check or lint catches it.
 *
 * Definitions are pooled across files, so a token scoped to one component
 * counts as defined everywhere. A floor, not a proof: it catches names that
 * exist nowhere.
 */

/**
 * `--name:` — a declaration in a stylesheet, or a key in an inline style
 * binding. A reference never has a colon after the name (`var(--name)` closes
 * with `)`, `var(--name, x)` with `,`), so the colon alone tells them apart.
 */
const DEFINITION = /(--[a-z0-9-]+)\s*['"]?\s*:/g
/** Every `var(--name` occurrence, including ones nested inside a fallback. */
const REFERENCE = /var\(\s*(--[a-z0-9-]+)/g
/** CSS/JS block comments and HTML comments. */
const COMMENT = /\/\*[\s\S]*?\*\/|<!--[\s\S]*?-->/g

export interface TokenUsage {
  defined: Set<string>
  referenced: Map<string, string[]>
}

/**
 * Blank out comments, keeping newlines so reported lines stay right. Tokens
 * named in prose would otherwise count as declarations.
 */
function stripComments(text: string): string {
  return text.replace(COMMENT, (comment) => comment.replace(/[^\n]/g, ' '))
}

function lineIndex(text: string): number[] {
  const starts = [0]
  for (let i = text.indexOf('\n'); i !== -1; i = text.indexOf('\n', i + 1)) starts.push(i + 1)
  return starts
}

function lineOf(starts: number[], offset: number): number {
  let low = 0
  let high = starts.length - 1
  while (low < high) {
    const mid = (low + high + 1) >> 1
    if (starts[mid] <= offset) low = mid
    else high = mid - 1
  }
  return low + 1
}

export function collectTokens(sources: Iterable<{ path: string; text: string }>): TokenUsage {
  const defined = new Set<string>()
  const referenced = new Map<string, string[]>()

  for (const source of sources) {
    const text = stripComments(source.text)
    for (const m of text.matchAll(DEFINITION)) defined.add(m[1])

    // Whole text, not per line: a `var(` whose name wraps a line must still match.
    const starts = lineIndex(text)
    for (const m of text.matchAll(REFERENCE)) {
      const sites = referenced.get(m[1]) ?? []
      sites.push(`${source.path}:${lineOf(starts, m.index ?? 0)}`)
      referenced.set(m[1], sites)
    }
  }
  return { defined, referenced }
}

export function undefinedTokens(usage: TokenUsage): Map<string, string[]> {
  const missing = new Map<string, string[]>()
  for (const [name, sites] of usage.referenced) {
    if (!usage.defined.has(name)) missing.set(name, sites)
  }
  return missing
}
