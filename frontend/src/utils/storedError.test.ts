import assert from 'node:assert/strict'
import test from 'node:test'

import { storedErrorOutput } from './storedError.ts'

test('a failed cell comes back with what it failed with', () => {
  const output = storedErrorOutput({ status: 'error', error: 'trimmed', error_full: 'full' })
  assert.equal(output?.error, 'trimmed')
  assert.equal(output?.traceback, 'trimmed')
  assert.equal(output?.tracebackFull, 'full')
})

test('a failure with no full traceback offers none', () => {
  const output = storedErrorOutput({ status: 'error', error: 'timed out', error_full: null })
  assert.equal(output?.tracebackFull, undefined)
})

test('only a cell that reads failed claims the failure', () => {
  assert.equal(storedErrorOutput({ status: 'running', error: 'old' }), undefined)
  assert.equal(storedErrorOutput({ status: 'stale', error: 'old' }), undefined)
  assert.equal(storedErrorOutput({ status: 'error', error: null }), undefined)
  assert.equal(storedErrorOutput({ status: 'error', error: '' }), undefined)
})
