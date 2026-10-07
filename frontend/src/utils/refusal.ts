// A request the server refused for want of permission (an `error` frame with
// code `read_only` or `insufficient_scope`): what the notebook says, so a
// refused run is not a click that silently did nothing. The server's own
// sentence is used as it is, since a deployment in front of Strata (a
// gateway admitting reads only) writes it for its members.

const REFUSALS = new Set(['read_only', 'insufficient_scope'])

/** The sentence to show for a refused frame, or `null` for any other error. */
export function refusalNotice(payload: unknown): string | null {
  if (!payload || typeof payload !== 'object') return null
  const { code, error } = payload as { code?: unknown; error?: unknown }
  if (typeof code !== 'string' || !REFUSALS.has(code)) return null
  if (typeof error === 'string' && error.trim()) return error.trim()
  return code === 'read_only'
    ? 'This notebook is read-only here: it can be opened and read, not run or changed.'
    : 'You do not have permission to do that in this notebook.'
}

/** The code of a refused REST call (`{"detail": {"error": "writes_disabled"}}`), or `null`. */
export function apiErrorCode(err: unknown): string | null {
  const payload = (err as { payload?: unknown } | null)?.payload
  const detail = (payload as { detail?: unknown } | null)?.detail
  const code = (detail as { error?: unknown } | null)?.error
  return typeof code === 'string' ? code : null
}
