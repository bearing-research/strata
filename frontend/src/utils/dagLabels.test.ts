import assert from 'node:assert/strict'
import test from 'node:test'

import { LABEL_CHAR_WIDTH, LABEL_HEIGHT, placeEdgeLabels } from './dagLabels.ts'

function assertNoOverlap(spots: ({ x: number; y: number } | null)[], text: string) {
  const width = text.length * LABEL_CHAR_WIDTH
  const shown = spots.filter((s) => s !== null)
  for (const [i, a] of shown.entries()) {
    for (const b of shown.slice(i + 1)) {
      const apart =
        a.x + width <= b.x ||
        b.x + width <= a.x ||
        a.y + LABEL_HEIGHT <= b.y ||
        b.y + LABEL_HEIGHT <= a.y
      assert.ok(apart, `labels overlap at ${JSON.stringify([a, b])}`)
    }
  }
}

test('a label with room stays at its edge', () => {
  assert.deepEqual(
    placeEdgeLabels(
      [
        { x: 0, y: 0, text: 'sales' },
        { x: 200, y: 0, text: 'sales' },
      ],
      [],
    ),
    [
      { x: 0, y: 0 },
      { x: 200, y: 0 },
    ],
  )
})

test('labels that land together move apart', () => {
  const spots = placeEdgeLabels(
    [
      { x: 10, y: 10, text: 'sales' },
      { x: 12, y: 11, text: 'sales' },
    ],
    [],
  )
  assert.deepEqual(spots[0], { x: 10, y: 10 })
  assert.ok(spots[1])
  assertNoOverlap(spots, 'sales')
})

test('with every spot taken, a label is dropped rather than drawn over another', () => {
  const crowd = Array.from({ length: 8 }, () => ({ x: 50, y: 50, text: 'sales' }))
  const spots = placeEdgeLabels(crowd, [])
  assert.equal(spots.at(-1), null)
  assertNoOverlap(spots, 'sales')
})

test('a label does not sit on a node', () => {
  const [spot] = placeEdgeLabels(
    [{ x: 10, y: 10, text: 'sales' }],
    [{ x: 0, y: 0, width: 100, height: 30 }],
  )
  assert.notDeepEqual(spot, { x: 10, y: 10 })
  assert.ok(spot)
  assert.ok(spot.y >= 30 || spot.y + LABEL_HEIGHT <= 0 || spot.x + 5 * LABEL_CHAR_WIDTH <= 0)
})
