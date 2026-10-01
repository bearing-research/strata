import assert from 'node:assert/strict'
import { test } from 'node:test'

import { shouldAdoptRemoteSource } from './cellSourceSync.ts'

const base = { remote: 'new', local: 'old', isDirty: false }

test('adopts a remote edit on a settled cell', () => {
  // An agent editing the cell over the CLI or MCP must reach the open editor.
  assert.equal(shouldAdoptRemoteSource(base), true)
})

test('ignores a snapshot that repeats what the editor already shows', () => {
  assert.equal(shouldAdoptRemoteSource({ ...base, remote: 'old' }), false)
})

test('ignores a payload with no source field', () => {
  assert.equal(shouldAdoptRemoteSource({ ...base, remote: undefined }), false)
  assert.equal(shouldAdoptRemoteSource({ ...base, remote: null }), false)
  assert.equal(shouldAdoptRemoteSource({ ...base, remote: 42 }), false)
})

test('never overwrites unflushed keystrokes', () => {
  // A snapshot arriving before the 2s idle flush must not discard typing.
  assert.equal(shouldAdoptRemoteSource({ ...base, isDirty: true }), false)
})

test('resumes adopting once the local edit is flushed', () => {
  // After a flushed local edit, remote edits must still be followed.
  assert.equal(shouldAdoptRemoteSource({ remote: 'theirs', local: 'mine', isDirty: false }), true)
})
