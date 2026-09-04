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
  // The window this protects: the user types, and a snapshot triggered by
  // something else (an agent running a different cell) arrives before the 2s
  // idle flush. Adopting here would silently discard what they typed.
  assert.equal(shouldAdoptRemoteSource({ ...base, isDirty: true }), false)
})

test('resumes adopting once the local edit is flushed', () => {
  // The regression an earlier in-flight hold introduced: after a local edit,
  // remote edits were ignored for the life of the page because the hold was
  // waiting on an echo that never came.
  assert.equal(shouldAdoptRemoteSource({ remote: 'theirs', local: 'mine', isDirty: false }), true)
})
