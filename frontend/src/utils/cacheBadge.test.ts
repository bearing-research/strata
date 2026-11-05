import assert from 'node:assert/strict'
import test from 'node:test'

import {
  cacheBadgeLabel,
  cacheBadgeTitle,
  cachedAgainFields,
  cacheFieldsAfterSync,
  durationLabel,
  formatDuration,
  isCachedAgain,
  restoredTimes,
} from './cacheBadge.ts'

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

test('only stale to ready with no run in between is cached again', () => {
  assert.equal(isCachedAgain('stale', 'ready'), true)
  assert.equal(isCachedAgain('running', 'ready'), false)
  assert.equal(isCachedAgain('idle', 'ready'), false)
  assert.equal(isCachedAgain('ready', 'ready'), false)
  assert.equal(isCachedAgain('stale', 'idle'), false)
  assert.equal(isCachedAgain(undefined, 'ready'), false)
})

test('a computed result that is current again saved what its run cost', () => {
  assert.deepEqual(cachedAgainFields({ cacheHit: false }, 2100), {
    cacheHit: true,
    cacheSavedMs: 2100,
  })
})

test('a hit that is current again keeps its saving, not its load time', () => {
  assert.deepEqual(cachedAgainFields({ cacheHit: true, cacheSavedMs: 2100 }, 11), {
    cacheHit: true,
    cacheSavedMs: 2100,
  })
})

test('a sync keeps a hit, what it saved and what it took to load', () => {
  const hit = { cacheHit: true, cacheSavedMs: 2100, cacheLoadMs: 11 }
  assert.deepEqual(cacheFieldsAfterSync(true, 'ready', hit, 11), hit)
})

test('a sync that finds a stale result current again reads as a hit', () => {
  assert.deepEqual(cacheFieldsAfterSync(true, 'ready', { cacheHit: false }, 2100), {
    cacheHit: true,
    cacheSavedMs: 2100,
    cacheLoadMs: undefined,
  })
})

test('a stale result keeps its saving through a sync, for when it is current again', () => {
  const hit = { cacheHit: true, cacheSavedMs: 2100, cacheLoadMs: 11 }
  const stale = cacheFieldsAfterSync(false, 'stale', hit, 11)
  assert.deepEqual(cacheFieldsAfterSync(true, 'ready', stale, 11), hit)
})

test('a result the server did not get from the cache is not a hit', () => {
  const hit = { cacheHit: true, cacheSavedMs: 2100 }
  assert.deepEqual(cacheFieldsAfterSync(false, 'ready', hit, 9), {})
  assert.deepEqual(cacheFieldsAfterSync(false, 'stale', { cacheHit: false }, 9), {})
  assert.deepEqual(cacheFieldsAfterSync(false, 'idle', undefined, 9), {})
})

test('a hit found on a freshly loaded page says what the server recorded it saved', () => {
  const fields = cacheFieldsAfterSync(true, 'ready', undefined, 2700)
  assert.equal(cacheBadgeLabel(fields.cacheSavedMs, fields.cacheLoadMs), 'cached · saved 2.7s')
})

test('a hit the server recorded as saving nothing reads plain cached after a page load', () => {
  const fields = cacheFieldsAfterSync(true, 'ready', undefined, 0)
  assert.equal(cacheBadgeLabel(fields.cacheSavedMs, fields.cacheLoadMs), 'cached')
})

test('a hit found on a page with no earlier output says nothing of a saving', () => {
  assert.deepEqual(cacheFieldsAfterSync(true, 'ready', undefined, undefined), {
    cacheHit: true,
    cacheSavedMs: undefined,
    cacheLoadMs: undefined,
  })
})

test('a run says it ran and a hit says it loaded', () => {
  assert.equal(durationLabel(3700, {}), 'ran in 3.7s')
  assert.equal(durationLabel(36, { cacheHit: true, cacheLoadMs: 36 }), 'loaded in 36ms')
})

test('a stale cell back to ready does not pass the run off as a load', () => {
  assert.equal(durationLabel(3700, { cacheHit: true }), '')
})

test('no figure, no label', () => {
  assert.equal(durationLabel(null, null), '')
  assert.equal(durationLabel(0, {}), '')
})

test('a reloaded page gets back what the last run took or the last hit loaded in', () => {
  const ran = { last_got: { cache_hit: false }, last_duration_ms: 2100 }
  assert.deepEqual(restoredTimes({ ...ran, status: 'ready' }), { runMs: 2100 })
  assert.deepEqual(restoredTimes({ ...ran, status: 'stale' }), { runMs: 2100 })

  const hit = { status: 'ready', last_got: { cache_hit: true }, last_duration_ms: 12 }
  assert.deepEqual(restoredTimes(hit), { loadMs: 12 })
})

test('a run claims no figure once its result is gone', () => {
  const ran = { last_got: { cache_hit: false }, last_duration_ms: 2100 }
  assert.deepEqual(restoredTimes({ ...ran, status: 'idle' }), {})
  assert.deepEqual(restoredTimes({ ...ran, status: 'error' }), {})
  assert.deepEqual(restoredTimes({ status: 'ready' }), {})
})

test('a live hit keeps its own load time over the restored one', () => {
  const hit = { cacheHit: true, cacheSavedMs: 2000, cacheLoadMs: 11 }
  assert.equal(cacheFieldsAfterSync(true, 'ready', hit, 2000, 99).cacheLoadMs, 11)
  assert.equal(cacheFieldsAfterSync(true, 'ready', undefined, 2000, 99).cacheLoadMs, 99)
})
