import assert from 'node:assert/strict'
import test from 'node:test'

import { MIN_FIT_SCALE, panToCenter, viewBox, viewScale, type DagView } from './dagViewport.ts'

const view = (over: Partial<DagView> = {}): DagView => ({
  graphWidth: 400,
  graphHeight: 200,
  panelWidth: 800,
  panelHeight: 400,
  zoom: 1,
  panX: 0,
  panY: 0,
  ...over,
})

test('a graph that fits fills the panel', () => {
  assert.deepEqual(viewBox(view()), { x: 0, y: 0, width: 400, height: 200 })
})

test('a graph wider than the panel keeps a readable scale and shows its middle', () => {
  const v = view({ graphWidth: 1000, panelWidth: 400, panelHeight: 600 })
  assert.equal(viewScale(v), MIN_FIT_SCALE)
  const box = viewBox(v)
  assert.equal(box.width, 400 / MIN_FIT_SCALE)
  assert.equal(box.x + box.width / 2, 500)
})

test('a graph taller than the panel starts at its top', () => {
  const box = viewBox(view({ graphHeight: 2000, panelWidth: 400, panelHeight: 400 }))
  assert.equal(box.y, 0)
})

test('zoom scales from the fit, and a drag moves the graph by what the pointer moved', () => {
  const v = view({ zoom: 2 })
  assert.equal(viewScale(v), 4)
  const moved = viewBox({ ...v, panX: 40 })
  assert.equal(viewBox(v).x - moved.x, 10) // 40 screen px at 4x
})

test('panToCenter puts the point at the center of the view', () => {
  const v = view({ graphWidth: 1000, graphHeight: 800, panelWidth: 400, panelHeight: 300 })
  const pan = panToCenter(v, { x: 900, y: 700 })
  const box = viewBox({ ...v, ...pan })
  assert.ok(Math.abs(box.x + box.width / 2 - 900) < 1e-9)
  assert.ok(Math.abs(box.y + box.height / 2 - 700) < 1e-9)
})
