import assert from 'node:assert/strict'
import test from 'node:test'

import { cacheBadgeLabel, cacheBadgeTitle, formatDuration } from './cacheBadge.ts'

test('a hit with a known cost says what it saved', () => {
  assert.equal(cacheBadgeLabel(2100, 11), 'cached · saved 2.1s')
  assert.equal(cacheBadgeLabel(340, undefined), 'cached · saved 340ms')
  assert.equal(cacheBadgeLabel(58, 58), 'cached · saved 58ms')
})

test('a hit with no known cost never claims a saving of zero', () => {
  for (const unknown of [0, null, undefined]) {
    assert.equal(cacheBadgeLabel(unknown, 11), 'cached')
  }
})

test('a hit that loaded slower than the run did claims no saving', () => {
  assert.equal(cacheBadgeLabel(6, 206), 'cached')
})

test('the tooltip keeps the load time', () => {
  assert.equal(cacheBadgeTitle(11), 'Result loaded from cache in 11ms')
  assert.equal(cacheBadgeTitle(0), 'Result loaded from cache')
  assert.equal(cacheBadgeTitle(undefined), 'Result loaded from cache')
})

test('durations switch to seconds at one second', () => {
  assert.equal(formatDuration(999), '999ms')
  assert.equal(formatDuration(1000), '1.0s')
})
