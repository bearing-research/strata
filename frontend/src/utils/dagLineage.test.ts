import assert from 'node:assert/strict'
import test from 'node:test'

import { dagLineage, edgeOnLineage } from './dagLineage.ts'

const e = (from: string, to: string) => ({ from, to })
const sorted = (s: Set<string>) => [...s].sort()

// a -> b -> c -> d, a -> d skipping the middle, and x -> y to one side.
const EDGES = [e('a', 'b'), e('b', 'c'), e('c', 'd'), e('a', 'd'), e('x', 'y')]

test('a selection reaches everything upstream and downstream of it, and nothing else', () => {
  const { ancestors, descendants } = dagLineage('b', EDGES)
  assert.deepEqual(sorted(ancestors), ['a'])
  assert.deepEqual(sorted(descendants), ['c', 'd'])
})

test('a node with no relatives has an empty lineage', () => {
  const { ancestors, descendants } = dagLineage('z', EDGES)
  assert.equal(ancestors.size + descendants.size, 0)
})

test('a cycle ends, and does not count the selection as its own relative', () => {
  const { ancestors, descendants } = dagLineage('p', [e('p', 'q'), e('q', 'p')])
  assert.deepEqual(sorted(ancestors), ['q'])
  assert.deepEqual(sorted(descendants), ['q'])
})

test('only edges on a path through the selection are on its lineage', () => {
  const lineage = dagLineage('b', EDGES)
  const on = EDGES.filter((edge) => edgeOnLineage(edge, 'b', lineage))
  assert.deepEqual(on, [e('a', 'b'), e('b', 'c'), e('c', 'd')])
})

test('an edge between ancestors of the selection is on its lineage', () => {
  const lineage = dagLineage('d', EDGES)
  const on = EDGES.filter((edge) => edgeOnLineage(edge, 'd', lineage))
  assert.deepEqual(on, [e('a', 'b'), e('b', 'c'), e('c', 'd'), e('a', 'd')])
})
