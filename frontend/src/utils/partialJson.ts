/**
 * Lenient parsing for incomplete JSON streams (issue #112).
 *
 * Prompt cells with `@output_schema` stream raw JSON over
 * `cell_output_delta` — mostly punctuation accumulating on one wrapped
 * line, which is unreadable as progress feedback. This module repairs a
 * truncated JSON prefix (close open strings/brackets, drop the dangling
 * tail) so the UI can pretty-print whatever is structurally complete and
 * let fields pop in as the model finishes them.
 *
 * The repair is best-effort by design: a prefix that can't be made
 * parseable within a bounded number of trim-and-retry passes returns
 * `undefined`, and the caller falls back to a char-count ticker. The
 * parsed value is transient display state — the canonical output is
 * still the final `cell_output` frame.
 */

interface ScanResult {
  /** Closing characters still owed, in opening order. */
  openStack: string[]
  /** True when the text ends inside a string literal. */
  inString: boolean
  /** Index of the last `,` outside any string, -1 when absent. */
  lastComma: number
  /** Index of the last `:` outside any string, -1 when absent. */
  lastColon: number
}

function scan(text: string): ScanResult {
  const openStack: string[] = []
  let inString = false
  let escaped = false
  let lastComma = -1
  let lastColon = -1

  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (inString) {
      if (escaped) {
        escaped = false
      } else if (c === '\\') {
        escaped = true
      } else if (c === '"') {
        inString = false
      }
      continue
    }
    if (c === '"') inString = true
    else if (c === '{') openStack.push('}')
    else if (c === '[') openStack.push(']')
    else if (c === '}' || c === ']') openStack.pop()
    else if (c === ',') lastComma = i
    else if (c === ':') lastColon = i
  }
  return { openStack, inString, lastComma, lastColon }
}

/** Close open strings/brackets and patch the dangling tail of a JSON prefix. */
function repair(text: string): string {
  const { openStack, inString } = scan(text)
  let out = text
  if (inString) out += '"'
  out = out.replace(/\s+$/, '')
  if (out.endsWith(',')) out = out.slice(0, -1)
  if (out.endsWith(':')) out += ' null'
  for (let i = openStack.length - 1; i >= 0; i--) out += openStack[i]
  return out
}

/**
 * Parse a (possibly truncated) JSON stream prefix.
 *
 * Returns the parsed value, or `undefined` when the prefix doesn't look
 * like JSON (must start with `{` or `[`) or can't be repaired. Trailing
 * partial tokens (`tru`, `1.2e`, a half-typed key) are dropped by
 * trimming back to the last structural comma/colon and re-repairing.
 */
export function parsePartialJson(text: string): unknown {
  let candidate = text.trimStart()
  if (!candidate.startsWith('{') && !candidate.startsWith('[')) return undefined

  // Bounded trim-and-retry: each pass cuts the unparseable tail back to
  // the previous structural separator. 32 passes is far beyond any
  // realistic nesting of dangling tokens between two deltas.
  for (let attempt = 0; attempt < 32; attempt++) {
    try {
      return JSON.parse(repair(candidate))
    } catch {
      const { lastComma, lastColon } = scan(candidate)
      const cut = Math.max(lastComma, lastColon > 0 ? lastColon + 1 : -1)
      if (cut <= 0 || cut >= candidate.length) return undefined
      candidate = candidate.slice(0, cut)
    }
  }
  return undefined
}

/** True when the stream buffer looks like a structured (JSON) response. */
export function isStructuredStream(buffer: string): boolean {
  const t = buffer.trimStart()
  return t.startsWith('{') || t.startsWith('[')
}
