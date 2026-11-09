import assert from 'node:assert/strict'
import test from 'node:test'

import { applyMountProgress, mountProgressLabel } from './mountProgress.ts'

test('a frame sets its mount and done drops it', () => {
  const one = applyMountProgress(undefined, { mount: 'raw', bytes_done: 0, bytes: 10 })
  const two = applyMountProgress(one, { mount: 'ref', bytes_done: 5, bytes: 9 })
  assert.deepEqual(two, { raw: { bytesDone: 0, bytes: 10 }, ref: { bytesDone: 5, bytes: 9 } })
  const left = applyMountProgress(two, { mount: 'raw', bytes_done: 10, bytes: 10, done: true })
  assert.deepEqual(left, { ref: { bytesDone: 5, bytes: 9 } })
  assert.equal(
    applyMountProgress(left, { mount: 'ref', bytes_done: 9, bytes: 9, done: true }),
    undefined,
  )
})

test('the label names the mount and both sizes', () => {
  assert.equal(
    mountProgressLabel('raw', { bytesDone: 3.2e9, bytes: 50e9 }),
    'hashing raw · 3.2 GB of 50.0 GB',
  )
})
