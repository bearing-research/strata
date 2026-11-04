import assert from 'node:assert/strict'
import test from 'node:test'

import { storedNotInRecents } from './storedNotebooks.ts'

test('a notebook already in the recents is listed once', () => {
  const stored = [{ path: '/nb/a' }, { path: '/nb/b' }, { path: '/nb/c' }]
  assert.deepEqual(storedNotInRecents(stored, [{ path: '/nb/b' }]), [
    { path: '/nb/a' },
    { path: '/nb/c' },
  ])
})

test('with no recents the storage directory is listed whole, in its order', () => {
  const stored = [{ path: '/nb/b' }, { path: '/nb/a' }]
  assert.deepEqual(storedNotInRecents(stored, []), stored)
})

test('a recent from elsewhere does not hide or add anything', () => {
  assert.deepEqual(storedNotInRecents([{ path: '/nb/a' }], [{ path: '/other/x' }]), [
    { path: '/nb/a' },
  ])
})
