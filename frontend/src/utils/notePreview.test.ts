import assert from 'node:assert/strict'
import test from 'node:test'

import { notePreview } from './notePreview.ts'

test('a heading of any level is skipped', () => {
  assert.equal(notePreview('# One\n\nBody'), 'Body')
  assert.equal(notePreview('## Step 2: run it again\n\nPress Run All'), 'Press Run All')
  assert.equal(notePreview('### Deep\nBody'), 'Body')
})

test('a note with only a heading has no preview', () => {
  assert.equal(notePreview('## Only a heading\n'), '')
})

test('text with no heading is its own preview', () => {
  assert.equal(notePreview('\n  Plain text\nMore'), 'Plain text')
})

test('a hash with no space is text', () => {
  assert.equal(notePreview('#hashtag'), '#hashtag')
})
