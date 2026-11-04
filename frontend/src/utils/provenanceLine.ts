import { formatDuration } from './cacheBadge.ts'

export interface LastGot {
  cacheHit: boolean
  /** What a hit saved (0 when nothing is known to have been saved), or what a run took. */
  ms: number
}

/** From a cell's output frame. A hit that took longer to load than the run did saved nothing. */
export function lastGotOf(
  cacheHit: boolean | undefined,
  savedMs: number | null | undefined,
  durationMs: number | null | undefined,
): LastGot {
  if (!cacheHit) return { cacheHit: false, ms: durationMs ?? 0 }
  return { cacheHit: true, ms: savedMs && savedMs >= (durationMs ?? 0) ? savedMs : 0 }
}

function howGot(got: LastGot): string {
  if (!got.cacheHit) return `took ${formatDuration(got.ms)}`
  return got.ms ? `from cache, saved ${formatDuration(got.ms)}` : 'from cache'
}

/** The quiet line under an output in the app view; empty for a cell nothing is known about. */
export function provenanceLine(
  computedAt: number | undefined,
  got: LastGot | undefined,
  formatTime: (epochSeconds: number) => string,
): string {
  const parts: string[] = []
  if (computedAt !== undefined) parts.push(`Computed ${formatTime(computedAt)}`)
  if (got) parts.push(howGot(got))
  return parts.join(' · ')
}
