import assert from 'node:assert/strict'
import test from 'node:test'

import { mergeReaders, readerLabel } from './datasetReaders.ts'

const here = { notebook_id: 'nb-here', cell_id: 'c1', reference: 'taxi/model@champion' }
const there = { notebook_id: 'nb-elsewhere-1234', cell_id: 'c9', reference: 'taxi/model' }

test('a cell the registry and this notebook both list appears once', () => {
  const recorded = { ...here, reference: 'taxi/model@v=1' }
  assert.deepEqual(mergeReaders([recorded, there], [here]), [here, there])
})

test('the same cell id in another notebook is another reader', () => {
  const sameCell = { ...there, cell_id: 'c1' }
  assert.deepEqual(mergeReaders([sameCell], [here]), [here, sameCell])
})

test('a cell here reads by its name, one elsewhere with its notebook', () => {
  assert.equal(readerLabel(here, 'nb-here', { c1: 'score' }), 'score')
  assert.equal(readerLabel(here, 'nb-here', {}), 'c1')
  assert.equal(readerLabel(there, 'nb-here', { c9: 'unrelated' }), 'nb-elsew/c9')
})
