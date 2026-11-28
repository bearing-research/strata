import assert from 'node:assert/strict'
import test from 'node:test'

import { heightTracker, MAX_HTML_OUTPUT_HEIGHT, reportedHeight } from './htmlOutput.ts'

const frame = {} as Window
const other = {} as Window

test('takes the height its own frame reports, rounded up', () => {
  assert.equal(reportedHeight({ source: frame, data: { strataOutputHeight: 466.2 } }, frame), 467)
})

test('ignores any other window, such as another output or a popup', () => {
  assert.equal(reportedHeight({ source: other, data: { strataOutputHeight: 300 } }, frame), null)
})

test('ignores a message that is not a usable height', () => {
  for (const data of [
    null,
    'x',
    {},
    { strataOutputHeight: '300' },
    { strataOutputHeight: 0 },
    { strataOutputHeight: Infinity },
  ]) {
    assert.equal(reportedHeight({ source: frame, data }, frame), null)
  }
})

test('caps a page that reports itself taller than any output should be', () => {
  assert.equal(
    reportedHeight({ source: frame, data: { strataOutputHeight: 1e9 } }, frame),
    MAX_HTML_OUTPUT_HEIGHT,
  )
})

test('has nothing to compare with before the frame exists', () => {
  assert.equal(reportedHeight({ source: frame, data: { strataOutputHeight: 300 } }, null), null)
})

test('follows a page that settles at its own height', () => {
  const next = heightTracker()
  assert.deepEqual([177, 466, 470, 300].map(next), [177, 466, 470, 300])
})

test('stops growing a page that grows with its frame', () => {
  const next = heightTracker()
  // 100vh plus 16px of body margin, from the frame's 150px start.
  assert.deepEqual([166, 182, 198, 214].map(next), [166, 182, null, null])
})
