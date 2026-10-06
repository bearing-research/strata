import assert from 'node:assert/strict'
import { test } from 'node:test'

import { DirtySources, shouldAdoptRemoteSource } from './cellSourceSync.ts'

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

// A tracker over a fake socket: `online` says whether frames go out, `sent` records them.
function tracker() {
  const link = { online: true, sent: [] as string[] }
  const dirty = new DirtySources((cellId) => {
    if (!link.online) return false
    link.sent.push(cellId)
    return true
  })
  return { link, dirty }
}

test('a flushed cell is clean', () => {
  const { link, dirty } = tracker()
  dirty.mark('a')
  dirty.flushAll()
  assert.deepEqual(link.sent, ['a'])
  assert.equal(dirty.has('a'), false)
})

test('an edit flushed while the socket is down stays dirty and goes out on the next flush', () => {
  const { link, dirty } = tracker()
  link.online = false
  dirty.mark('a')
  dirty.flushAll()
  // Still dirty, so the reconnect sync cannot replace the buffer.
  assert.equal(dirty.has('a'), true)
  assert.equal(
    shouldAdoptRemoteSource({ remote: 'server', local: 'typed', isDirty: dirty.has('a') }),
    false,
  )

  link.online = true
  dirty.flushAll()
  assert.deepEqual(link.sent, ['a'])
  assert.equal(dirty.has('a'), false)
})

test('an edit refused while its cell runs is resent when the run ends', () => {
  const { link, dirty } = tracker()
  dirty.mark('a')
  dirty.flush('a')
  dirty.refusedWhileRunning('a')
  assert.equal(dirty.has('a'), true)

  dirty.statusChanged('a', 'running')
  assert.deepEqual(link.sent, ['a'], 'not while it still runs')
  dirty.statusChanged('a', 'ready')
  assert.deepEqual(link.sent, ['a', 'a'])
  assert.equal(dirty.has('a'), false)

  // Only once: a later status does not resend.
  dirty.statusChanged('a', 'stale')
  assert.deepEqual(link.sent, ['a', 'a'])
})

test('a status change does not flush typing that was never refused', () => {
  const { link, dirty } = tracker()
  dirty.mark('a')
  dirty.statusChanged('a', 'ready')
  assert.deepEqual(link.sent, [])
  assert.equal(dirty.has('a'), true)
})
