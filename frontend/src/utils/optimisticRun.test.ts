import assert from 'node:assert/strict'
import test from 'node:test'

import { OptimisticRuns } from './optimisticRun.ts'

const busy = (cellId: string) => ({
  error: 'Notebook is already executing cell a',
  code: 'notebook_busy',
  cell_id: cellId,
})

test('a busy refusal puts the cell back to what it showed before the click', () => {
  const runs = new OptimisticRuns()
  runs.start('b', 'ready')
  assert.deepEqual(runs.refused(busy('b')), { cellId: 'b', status: 'ready' })
  // Once: a repeated frame restores nothing more.
  assert.equal(runs.refused(busy('b')), null)
})

test('a status from the server wins over the rollback', () => {
  const runs = new OptimisticRuns()
  runs.start('b', 'ready')
  runs.settle('b')
  assert.equal(runs.refused(busy('b')), null)
})

test('a second click keeps the status from before the first', () => {
  const runs = new OptimisticRuns()
  runs.start('b', 'stale')
  runs.start('b', 'running')
  assert.deepEqual(runs.refused(busy('b')), { cellId: 'b', status: 'stale' })
})

test('other errors and other cells are left alone', () => {
  const runs = new OptimisticRuns()
  runs.start('b', 'ready')
  assert.equal(runs.refused({ error: 'x', code: 'cell_busy', cell_id: 'b' }), null)
  assert.equal(runs.refused({ error: 'x', code: 'notebook_busy' }), null)
  assert.equal(runs.refused(busy('c')), null)
  assert.equal(runs.refused(null), null)
  assert.deepEqual(runs.refused(busy('b')), { cellId: 'b', status: 'ready' })
})
