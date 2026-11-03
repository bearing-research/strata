import assert from 'node:assert/strict'
import test from 'node:test'

import { cellDotTitle, cellRunBadge } from './cellRunBadge.ts'

test('a stale cell says stale, however its last run got its result', () => {
  assert.equal(cellRunBadge('stale', true), 'stale')
  assert.equal(cellRunBadge('stale', false), 'stale')
  assert.equal(cellRunBadge('stale', undefined), 'stale')
})

test('a current cell shows its cache state, and no badge without a hit', () => {
  assert.equal(cellRunBadge('ready', true), 'cached')
  assert.equal(cellRunBadge('ready', false), null)
  assert.equal(cellRunBadge('idle', undefined), null)
})

const plain = { isLeaf: false, executionMethod: '', hasRunBadge: false }

test('the dot says the status alone when there is nothing more', () => {
  assert.equal(cellDotTitle('ready', plain), 'ready')
})

test('the dot carries the leaf and execution-method labels', () => {
  assert.equal(
    cellDotTitle('ready', { ...plain, isLeaf: true, executionMethod: 'batch' }),
    'ready\nLeaf: no downstream consumers\nExecuted via batch process',
  )
})

test('the dot does not describe a run the cell badge already replaced', () => {
  assert.equal(
    cellDotTitle('stale · upstream changed', {
      ...plain,
      executionMethod: 'batch',
      hasRunBadge: true,
    }),
    'stale · upstream changed',
  )
})
