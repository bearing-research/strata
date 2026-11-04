import assert from 'node:assert/strict'
import test from 'node:test'

import { outputLabel } from './storeRows.ts'

test('a variable reads as itself, with its loop or variant suffix', () => {
  assert.equal(outputLabel('df', 'x'), 'df')
  assert.equal(outputLabel('state@iter=2', 'x'), 'state@iter=2')
})

test('reserved records read as what they are', () => {
  assert.equal(outputLabel('__display__0', 'x'), 'display 0')
  assert.equal(outputLabel('__display__12@variant=a', 'x'), 'display 12')
  assert.equal(outputLabel('__console__', 'x'), 'console')
  assert.equal(outputLabel(`__file__${'a'.repeat(64)}`, 'x'), 'file')
})

test('a row no cell produced falls back to its id', () => {
  assert.equal(outputLabel(null, 'some_id'), 'some_id')
  assert.equal(outputLabel(undefined, 'some_id'), 'some_id')
})
