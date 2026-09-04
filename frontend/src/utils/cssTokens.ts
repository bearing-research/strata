/**
 * Collecting the CSS custom properties the frontend defines and references.
 *
 * A `var(--typo)` fails silently: with a fallback the declaration quietly uses
 * a hardcoded literal, and without one it is dropped and the property inherits.
 * Neither shows up in a build, a type check, or a lint pass. That is how the
 * lineage modal ended up painting `var(--bg, #fff)` — pure white — underneath
 * theme-coloured text, unreadable in dark mode and fine in light mode by
 * accident.
 */

/**
 * `--name:` — a declaration in a stylesheet, or a key in an inline style
 * binding. A reference never has a colon after the name (`var(--name)` closes
 * with `)`, `var(--name, x)` with `,`), so the colon alone tells them apart.
 */
const DEFINITION = /(--[a-z0-9-]+)\s*['"]?\s*:/g
/** Every `var(--name` occurrence, including ones nested in a fallback. */
const REFERENCE = /var\(\s*(--[a-z0-9-]+)/g

export interface TokenUsage {
  defined: Set<string>
  referenced: Map<string, string[]>
}

export function collectTokens(sources: Iterable<{ path: string; text: string }>): TokenUsage {
  const defined = new Set<string>()
  const referenced = new Map<string, string[]>()

  for (const { path, text } of sources) {
    for (const m of text.matchAll(DEFINITION)) defined.add(m[1])
    text.split('\n').forEach((line, i) => {
      for (const m of line.matchAll(REFERENCE)) {
        const sites = referenced.get(m[1]) ?? []
        sites.push(`${path}:${i + 1}`)
        referenced.set(m[1], sites)
      }
    })
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
