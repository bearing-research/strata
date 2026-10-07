import test from 'node:test'
import assert from 'node:assert/strict'
import {
  findRecentNotebookBySessionId,
  normalizeRecentNotebookEntries,
  recordRecentNotebookEntries,
  removeRecentNotebookEntries,
} from './recentNotebooks.ts'

test('normalizeRecentNotebookEntries keeps valid entries and orders newest first', () => {
  const normalized = normalizeRecentNotebookEntries([
    { name: 'older', path: '/tmp/older', lastOpened: 10, sessionId: 's1' },
    { name: 'newer', path: '/tmp/newer', lastOpened: 20, sessionId: 's2' },
    { name: 'invalid', path: 123, lastOpened: 5 },
  ])

  assert.deepEqual(normalized, [
    { name: 'newer', path: '/tmp/newer', lastOpened: 20, sessionId: 's2' },
    { name: 'older', path: '/tmp/older', lastOpened: 10, sessionId: 's1' },
  ])
})

test('recordRecentNotebookEntries deduplicates by path and updates session id', () => {
  const updated = recordRecentNotebookEntries(
    [
      { name: 'Notebook', path: '/tmp/notebook', lastOpened: 10, sessionId: 'old-session' },
      { name: 'Other', path: '/tmp/other', lastOpened: 5, sessionId: 'other-session' },
    ],
    'Notebook',
    '/tmp/notebook',
    'new-session',
    99,
  )

  assert.deepEqual(updated, [
    { name: 'Notebook', path: '/tmp/notebook', lastOpened: 99, sessionId: 'new-session' },
    { name: 'Other', path: '/tmp/other', lastOpened: 5, sessionId: 'other-session' },
  ])
})

test('findRecentNotebookBySessionId returns the matching notebook path', () => {
  const match = findRecentNotebookBySessionId(
    [
      { name: 'Notebook', path: '/tmp/notebook', lastOpened: 10, sessionId: 'session-a' },
      { name: 'Other', path: '/tmp/other', lastOpened: 5, sessionId: 'session-b' },
    ],
    'session-b',
  )

  assert.deepEqual(match, {
    name: 'Other',
    path: '/tmp/other',
    lastOpened: 5,
    sessionId: 'session-b',
  })
})

test('removeRecentNotebookEntries removes only the matching path', () => {
  const updated = removeRecentNotebookEntries(
    [
      { name: 'Notebook', path: '/tmp/notebook', lastOpened: 10, sessionId: 'session-a' },
      { name: 'Other', path: '/tmp/other', lastOpened: 5, sessionId: 'session-b' },
    ],
    '/tmp/notebook',
  )

  assert.deepEqual(updated, [
    { name: 'Other', path: '/tmp/other', lastOpened: 5, sessionId: 'session-b' },
  ])
})

test('blocked site storage leaves recents empty and keeps recording in memory', async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    get() {
      throw new DOMException('Access is denied for this document.', 'SecurityError')
    },
  })
  try {
    // A fresh module instance, so its load() runs under the throwing getter.
    const mod = await import('./recentNotebooks.ts?blocked-storage')
    const recents = mod.useRecentNotebooks()
    assert.deepEqual(recents.entries.value, [])
    recents.record('nb', '/tmp/nb', 's1')
    assert.equal(recents.entries.value[0]?.path, '/tmp/nb')
  } finally {
    if (original) Object.defineProperty(globalThis, 'localStorage', original)
    else delete (globalThis as { localStorage?: unknown }).localStorage
  }
})
