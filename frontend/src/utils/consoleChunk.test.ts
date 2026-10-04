import assert from 'node:assert/strict'
import test from 'node:test'

import { applyConsoleChunk } from './consoleChunk.ts'

test("a run's first chunk replaces the last run's console", () => {
  assert.equal(applyConsoleChunk('epoch 9 of the last run\n', 'epoch 1\n', 0), 'epoch 1\n')
})

test('later chunks append', () => {
  assert.equal(applyConsoleChunk('epoch 1\n', 'epoch 2\n', 1), 'epoch 1\nepoch 2\n')
})

test('console sent when the cell finishes appends', () => {
  assert.equal(applyConsoleChunk('epoch 1\n', 'done\n', null), 'epoch 1\ndone\n')
  assert.equal(applyConsoleChunk(undefined, 'done\n', undefined), 'done\n')
})
