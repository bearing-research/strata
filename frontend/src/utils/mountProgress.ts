import { formatBytes } from './inputWeights.ts'

/** How much of one local mount a run or the background check has hashed. */
export interface MountProgress {
  bytesDone: number
  bytes: number
}

/** A cell's mounts being hashed after one ``cell_mount_progress`` frame; ``done`` drops one. */
export function applyMountProgress(
  current: Record<string, MountProgress> | undefined,
  frame: { mount: string; bytes_done: number; bytes: number; done?: boolean },
): Record<string, MountProgress> | undefined {
  const next = { ...current }
  if (frame.done) {
    delete next[frame.mount]
  } else {
    next[frame.mount] = { bytesDone: frame.bytes_done, bytes: frame.bytes }
  }
  return Object.keys(next).length ? next : undefined
}

export function mountProgressLabel(mount: string, progress: MountProgress): string {
  return `hashing ${mount} · ${formatBytes(progress.bytesDone)} of ${formatBytes(progress.bytes)}`
}
