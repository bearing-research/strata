import assert from 'node:assert/strict'
import test from 'node:test'

import { cellRunBadge } from './cellRunBadge.ts'

test('a stale cell says stale, however its last run got its result', () => {
  assert.equal(cellRunBadge('stale', true, ''), 'stale')
  assert.equal(cellRunBadge('stale', true, 'cached'), 'stale')
  assert.equal(cellRunBadge('stale', false, 'warm'), 'stale')
  assert.equal(cellRunBadge('stale', undefined, ''), 'stale')
})

test('a current cell keeps its cache or execution-method badge', () => {
  assert.equal(cellRunBadge('ready', true, 'cached'), 'cached')
  assert.equal(cellRunBadge('ready', false, 'warm'), 'method')
  assert.equal(cellRunBadge('error', undefined, 'cold'), 'method')
})

test('a cell with no run to describe has no badge', () => {
  assert.equal(cellRunBadge('idle', undefined, ''), null)
})
