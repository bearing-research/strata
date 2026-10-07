// A connection's top-level keys the form does not edit round-trip unchanged.
// Empty ones (the server's `options = {}` and `credential = null` defaults) are
// left out: there is nothing in them to preserve, and listing them would claim
// every connection carries extra settings.

function isEmpty(value: unknown): boolean {
  if (value === null || value === undefined) return true
  if (Array.isArray(value)) return value.length === 0
  if (typeof value === 'object') return Object.keys(value).length === 0
  return false
}

export function connectionExtras(
  spec: Record<string, unknown>,
  knownKeys: ReadonlySet<string>,
): Record<string, unknown> {
  const extras: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(spec)) {
    if (knownKeys.has(key) || isEmpty(value)) continue
    extras[key] = value
  }
  return extras
}
