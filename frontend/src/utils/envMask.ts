// The server never sends a secret env value: it sends this marker instead, and
// the marker sent back on save means "keep the current value".
export const MASKED_ENV_VALUE = '__strata_masked__'

export interface EnvDraftRow {
  key: string
  value: string
  // The key the server sent this row's value masked under, else null.
  maskedKey: string | null
}

export function envDraftRows(env: Record<string, string>): EnvDraftRow[] {
  return Object.entries(env).map(([key, value]) =>
    value === MASKED_ENV_VALUE
      ? { key, value: '', maskedKey: key }
      : { key, value, maskedKey: null },
  )
}

/** A masked row left blank under its original name keeps its value. */
export function envPayload(rows: EnvDraftRow[]): Record<string, string> {
  return Object.fromEntries(
    rows
      .map((row) => {
        const key = row.key.trim()
        const unchanged = row.maskedKey !== null && row.value === '' && key === row.maskedKey
        return [key, unchanged ? MASKED_ENV_VALUE : row.value] as const
      })
      .filter(([key]) => key),
  )
}
