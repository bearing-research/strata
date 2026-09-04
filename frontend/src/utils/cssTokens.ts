/**
 * Collecting the CSS custom properties the frontend defines and references.
 *
 * A missing token fails silently: with a fallback the declaration quietly uses
 * a hardcoded literal, and without one it is dropped and the property
 * inherits. Neither shows up in a build, a type check, or a lint pass. That is
 * how the lineage modal ended up painting pure white underneath theme-coloured
 * text, unreadable in dark mode and fine in light mode by accident.
 *
 * Known limitation: definitions are pooled across every file, so a token
 * declared in one component's scoped style counts as defined for all of them.
 * Modelling which elements actually inherit a declaration would mean modelling
 * the DOM. The check is therefore a floor, not a proof: it catches names that
 * exist nowhere, which is the failure that has actually happened here.
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
 * Blank out comments, keeping every newline so reported lines stay right.
 *
 * Prose is not code: this file's own docstring names tokens, and without this
 * the checker would read them as declarations and consider them defined.
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

    // Scanned over the whole text rather than line by line: a `var(` whose
    // token name wraps to the next line would otherwise never match, and go
    // unchecked — the one thing this file exists to catch.
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
