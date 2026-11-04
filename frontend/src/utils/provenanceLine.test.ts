import assert from 'node:assert/strict'
import test from 'node:test'

import { lastGotOf, provenanceLine } from './provenanceLine.ts'

const when = () => 'Nov 4, 9:15 AM'

test('a run says when and how long it took', () => {
  assert.equal(
    provenanceLine(100, { cacheHit: false, ms: 2100 }, when),
    'Computed Nov 4, 9:15 AM · took 2.1s',
  )
})

test('a hit says what it saved', () => {
  assert.equal(
    provenanceLine(100, { cacheHit: true, ms: 2100 }, when),
    'Computed Nov 4, 9:15 AM · from cache, saved 2.1s',
  )
})

test('a hit with nothing known saved does not claim a saving', () => {
  assert.equal(provenanceLine(100, { cacheHit: true, ms: 0 }, when), 'Computed Nov 4, 9:15 AM · from cache')
})

test('a part that is not known is left out', () => {
  assert.equal(provenanceLine(undefined, { cacheHit: false, ms: 40 }, when), 'took 40ms')
  assert.equal(provenanceLine(100, undefined, when), 'Computed Nov 4, 9:15 AM')
  assert.equal(provenanceLine(undefined, undefined, when), '')
})

test('an output frame reads as a run or a hit, and a slow load saved nothing', () => {
  assert.deepEqual(lastGotOf(false, 500, 2100), { cacheHit: false, ms: 2100 })
  assert.deepEqual(lastGotOf(true, 2100, 40), { cacheHit: true, ms: 2100 })
  assert.deepEqual(lastGotOf(true, 30, 40), { cacheHit: true, ms: 0 })
  assert.deepEqual(lastGotOf(true, undefined, 40), { cacheHit: true, ms: 0 })
})
