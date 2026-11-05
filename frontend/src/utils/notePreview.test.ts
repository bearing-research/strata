import assert from 'node:assert/strict'
import test from 'node:test'

import { noteHeading, notePreview } from './notePreview.ts'

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

test('markup is stripped from a preview', () => {
  assert.equal(notePreview('Press **Run All** again.'), 'Press Run All again.')
  assert.equal(notePreview('Change `.sum()` to *mean*.'), 'Change .sum() to mean.')
  assert.equal(notePreview('See [the docs](https://example.com) now'), 'See the docs now')
  assert.equal(notePreview('![a chart](chart.png) above'), 'a chart above')
  assert.equal(notePreview('a \\*literal\\* star &amp; more'), 'a *literal* star & more')
})

test('a block marker is not part of the preview', () => {
  assert.equal(notePreview('- first item\n- second'), 'first item')
  assert.equal(notePreview('> quoted\n> more'), 'quoted')
  assert.equal(notePreview('1. **one**'), 'one')
})

test('only the first line of a paragraph is the preview', () => {
  assert.equal(notePreview('one **two**\nthree'), 'one two')
})

test('a note of only code previews its first line, not its fence', () => {
  assert.equal(notePreview('```python\nprint(1)\nprint(2)\n```'), 'print(1)')
})

test('a hash inside a fence is not a heading', () => {
  assert.equal(notePreview('```\n# a comment\n```'), '# a comment')
  assert.equal(noteHeading('```\n# a comment\n```'), null)
})

test('a heading is stripped of its markup', () => {
  assert.equal(noteHeading('# Step **2**: `run` it'), 'Step 2: run it')
  assert.equal(noteHeading('text\n\n## Later'), 'Later')
  assert.equal(noteHeading('Underlined\n==='), 'Underlined')
  assert.equal(notePreview('Underlined\n===\n\nBody'), 'Body')
})

test('a note with no heading has none', () => {
  assert.equal(noteHeading('just text'), null)
  assert.equal(noteHeading(''), null)
  assert.equal(notePreview(''), '')
})
