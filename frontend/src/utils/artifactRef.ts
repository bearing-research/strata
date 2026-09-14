/**
 * Artifact references as the backend writes them: `strata://artifact/<id>@v=<n>`.
 */

export interface ArtifactRef {
  id: string
  version: number
}

const PREFIX = 'strata://artifact/'

/** `strata://artifact/<id>@v=<n>` → `{id, version}`, or null for anything else. */
export function parseArtifactRef(uri: string): ArtifactRef | null {
  if (!uri.startsWith(PREFIX)) return null
  const ref = uri.slice(PREFIX.length)
  const at = ref.lastIndexOf('@v=')
  if (at <= 0) return null
  const version = Number(ref.slice(at + 3))
  if (!Number.isInteger(version) || version < 1) return null
  return { id: ref.slice(0, at), version }
}

/** A cell's `{var_name: uri}` map, keeping only entries that are artifact refs. */
export function parseArtifactUris(raw: unknown): Record<string, string> {
  if (!raw || typeof raw !== 'object') return {}
  const out: Record<string, string> = {}
  for (const [name, uri] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof uri === 'string' && parseArtifactRef(uri)) out[name] = uri
  }
  return out
}

/**
 * The outputs a cell can offer for promotion: only while it is ready, and only
 * variables it still defines. The backend keeps every variable a cell has ever
 * stored, so after a rename or an unrun edit the full map would offer an
 * outdated result under a team name.
 */
export function promotableOutputs(
  cell: { status: string; defines: string[]; artifactUris?: Record<string, string> } | undefined,
): Record<string, string> {
  if (!cell || cell.status !== 'ready') return {}
  return Object.fromEntries(
    Object.entries(cell.artifactUris || {}).filter(([name]) => cell.defines.includes(name)),
  )
}
