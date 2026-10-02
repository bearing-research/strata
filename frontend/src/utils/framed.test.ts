import assert from 'node:assert/strict'
import test from 'node:test'

import { editorChrome, editorQuery } from './framed.ts'

test('standalone, the editor has all its chrome', () => {
  assert.deepEqual(editorChrome({}), {
    brand: true,
    pageLinks: true,
    modeBadge: true,
    deleteNotebook: true,
  })
  assert.equal(editorChrome({ framed: '0' }).brand, true)
  assert.equal(editorChrome({ path: 'a/b' }).deleteNotebook, true)
})

test('framed, the parent has the brand, the links and delete', () => {
  for (const framed of ['1', 'true', ['1']]) {
    assert.deepEqual(editorChrome({ framed }), {
      brand: false,
      pageLinks: false,
      modeBadge: false,
      deleteNotebook: false,
    })
  }
})

test('reopening the editor keeps it framed, and only when it was', () => {
  assert.deepEqual(editorQuery({ framed: '1', path: 'old' }, 'nb/a'), { path: 'nb/a', framed: '1' })
  assert.deepEqual(editorQuery({ framed: ['true'] }, 'nb/a'), { path: 'nb/a', framed: '1' })
  assert.deepEqual(editorQuery({ path: 'old' }, 'nb/a'), { path: 'nb/a' })
  assert.deepEqual(editorQuery({ framed: '0' }, 'nb/a'), { path: 'nb/a' })
})
