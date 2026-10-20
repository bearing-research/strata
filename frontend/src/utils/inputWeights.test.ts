import assert from 'node:assert/strict'
import test from 'node:test'

import type { InputWeight } from '../types/ws-payloads.generated.ts'
import { formatBytes, inputWeightBadge } from './inputWeights.ts'

const mount: InputWeight = {
  kind: 'mount',
  name: 'data',
  uri: 's3://bucket/data',
  bytes: 3 * 1000 ** 3,
  cached_bytes: 1000 ** 3,
}

test('formatBytes uses decimal units, as the read thresholds do', () => {
  assert.deepEqual([0, 999, 1500, 5 * 1000 ** 2, 10 ** 9, 2.5 * 1000 ** 3].map(formatBytes), [
    '0 B',
    '999 B',
    '1.5 KB',
    '5.0 MB',
    '1.0 GB',
    '2.5 GB',
  ])
})

test('while it runs, the badge counts what is not already on the machine', () => {
  const table: InputWeight = {
    kind: 'table',
    name: 'events',
    uri: 'lake#db.events',
    bytes: 1000 ** 2,
  }

  assert.deepEqual(
    inputWeightBadge({ status: 'running', machine: 'local', inputs: [mount, table] }),
    {
      label: 'up to 2.0 GB to read',
      title: [
        'Reads on "local"',
        'mount data (s3://bucket/data): 3.0 GB, 1.0 GB already here',
        'table events (lake#db.events): up to 1.0 MB',
      ].join('\n'),
    },
  )
  assert.equal(
    inputWeightBadge({ status: 'running', machine: 'local', inputs: [mount] })?.label,
    '2.0 GB to read',
  )
})

test('at rest, or with nothing to move, the badge says what the inputs weigh', () => {
  const warm = { ...mount, cached_bytes: mount.bytes }
  const table: InputWeight = { kind: 'table', name: 'events', uri: 'lake#db.events', bytes: 1000 }

  assert.equal(
    inputWeightBadge({ status: 'ready', machine: 'local', inputs: [mount] })?.label,
    'reads 3.0 GB',
  )
  assert.equal(
    inputWeightBadge({ status: 'running', machine: 'local', inputs: [warm] })?.label,
    'reads 3.0 GB',
  )
  assert.equal(
    inputWeightBadge({ status: 'ready', machine: 'local', inputs: [mount, table] })?.label,
    'reads up to 3.0 GB',
  )
})

test('a share the machine did not report bounds what moves', () => {
  const remote = { ...mount, cached_bytes: null }

  assert.equal(
    inputWeightBadge({ status: 'running', machine: 'gpu', inputs: [remote] })?.label,
    'up to 3.0 GB to read',
  )
})

test('an unknown size is said, never guessed', () => {
  const pinned: InputWeight = { kind: 'mount', name: 'v1', uri: 's3://bucket/v1' }

  assert.equal(
    inputWeightBadge({ status: 'running', machine: 'local', inputs: [mount, pinned] })?.label,
    '2.0 GB to read + unknown',
  )
  assert.equal(
    inputWeightBadge({ status: 'ready', machine: 'local', inputs: [pinned] })?.label,
    'read size unknown',
  )
})

test('no badge for a cell that reads none', () => {
  assert.equal(inputWeightBadge({ status: 'running', machine: 'local', inputs: [] }), null)
})
