import type { InputWeight } from '../types/ws-payloads.generated'

// Decimal, as STRATA_NOTEBOOK_CONFIRM_READ_BYTES and the per-GB rates are.
export function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let unit = 0
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000
    unit += 1
  }
  return unit === 0 ? `${value} B` : `${value.toFixed(1)} ${units[unit]}`
}

export function describeInput(input: InputWeight): string {
  const source = `${input.kind} ${input.name} (${input.uri})`
  if (input.bytes == null) return `${source}: size unknown`
  // A table's size is its snapshot's data files; a scan may read less.
  const size =
    input.kind === 'table' ? `up to ${formatBytes(input.bytes)}` : formatBytes(input.bytes)
  if (input.cached_bytes == null) return `${source}: ${size}`
  return `${source}: ${size}, ${formatBytes(input.cached_bytes)} already here`
}

/**
 * The badge on a cell that reads mounts, tables or fetches: what they weigh, and while
 * it runs, how much of it moves onto the machine. What is not known is not guessed: a
 * table's size, or a share the machine did not report, makes a total an upper bound,
 * and an unknown size shows as such.
 */
export function inputWeightBadge(params: {
  status: string
  machine?: string | null
  inputs?: InputWeight[] | null
}): { label: string; title: string } | null {
  const { status, machine, inputs } = params
  if (!inputs?.length) return null
  let size = 0
  let toMove = 0
  let sizeBound = false
  let moveBound = false
  let unknown = false
  for (const input of inputs) {
    if (input.bytes == null) {
      unknown = true
      continue
    }
    size += input.bytes
    toMove += input.bytes - (input.cached_bytes ?? 0)
    sizeBound ||= input.kind === 'table'
    moveBound ||= input.cached_bytes == null
  }
  const more = unknown ? ' + unknown' : ''
  const upTo = (bound: boolean) => (bound ? 'up to ' : '')
  let label: string
  if (status === 'running' && toMove > 0) {
    label = `${upTo(moveBound)}${formatBytes(toMove)} to read${more}`
  } else if (size > 0 || !unknown) {
    label = `reads ${upTo(sizeBound)}${formatBytes(size)}${more}`
  } else {
    label = 'read size unknown'
  }
  const title = [`Reads on "${machine}"`, ...inputs.map(describeInput)].join('\n')
  return { label, title }
}
