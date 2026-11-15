import assert from 'node:assert/strict'
import test from 'node:test'

import { branchOrigin, checkpointContents, parseCheckpointRuns } from './checkpoints.ts'

const checkpoint = {
  release_id: 'r2',
  sequence: 2,
  created_at: 1700000002.5,
  parts: ['metrics', 'model'],
  files: 0,
  step: null,
  byte_size: 43,
  heads: ['best'],
}
const trunk = {
  build_id: 'b1',
  status: 'finished',
  variant: null,
  init: null,
  started_at: 1700000000,
  checkpoints: [checkpoint, { ...checkpoint, release_id: 'r1', sequence: 1, heads: [] }],
}
const branch = {
  build_id: 'b2',
  status: 'running',
  variant: 'lr=0.1',
  init: 'r1',
  started_at: 1700000100,
  checkpoints: [{ ...checkpoint, release_id: 'r3', sequence: 1, files: 2, step: 7, parts: [] }],
}

test('the listing arrives as the server sent it, under the names the panel uses', () => {
  const [parsedBranch, parsedTrunk] = parseCheckpointRuns([branch, trunk])

  assert.deepEqual(parsedTrunk, {
    buildId: 'b1',
    status: 'finished',
    variant: null,
    init: null,
    startedAt: 1700000000,
    checkpoints: [
      {
        releaseId: 'r2',
        sequence: 2,
        createdAt: 1700000002.5,
        parts: ['metrics', 'model'],
        files: 0,
        step: null,
        byteSize: 43,
        heads: ['best'],
      },
      {
        releaseId: 'r1',
        sequence: 1,
        createdAt: 1700000002.5,
        parts: ['metrics', 'model'],
        files: 0,
        step: null,
        byteSize: 43,
        heads: [],
      },
    ],
  })
  assert.equal(parsedBranch.variant, 'lr=0.1')
  assert.equal(parsedBranch.init, 'r1')
  assert.equal(parsedBranch.checkpoints[0].step, 7)
})

test('a run whose row is gone, or in a state the panel has no word for, has no status', () => {
  const [gone, odd] = parseCheckpointRuns([
    { ...trunk, status: null, started_at: null },
    { ...trunk, status: 'pending' },
  ])

  assert.equal(gone.status, null)
  assert.equal(gone.startedAt, null)
  assert.equal(odd.status, null)
})

test('a checkpoint is described by its parts, and its files by how many', () => {
  const [parsedBranch, parsedTrunk] = parseCheckpointRuns([branch, trunk])

  assert.equal(checkpointContents(parsedTrunk.checkpoints[0]), 'metrics, model')
  assert.equal(checkpointContents(parsedBranch.checkpoints[0]), '2 files')
  assert.equal(
    checkpointContents({ ...parsedBranch.checkpoints[0], files: 1, parts: ['epoch'] }),
    'epoch, 1 file',
  )
})

test('a branch names the checkpoint it started from while that is still listed', () => {
  const runs = parseCheckpointRuns([branch, trunk])

  const origin = branchOrigin(runs[0], runs)

  assert.equal(origin?.run.buildId, 'b1')
  assert.equal(origin?.checkpoint.sequence, 1)
  assert.equal(branchOrigin(runs[1], runs), null)
  // Collected since: the branch is still listed, its origin is not.
  assert.equal(branchOrigin({ ...runs[0], init: 'gone' }, runs), null)
})
