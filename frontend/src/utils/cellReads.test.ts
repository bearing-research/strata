import assert from 'node:assert/strict'
import test from 'node:test'

import { cellReads } from './cellReads.ts'

const edge = (to_cell_id: string, variable: string) => ({ to_cell_id, variable })

test('reads lists what the incoming edges carry, in the order the cell refers to them', () => {
  const edges = [edge('b', 'model'), edge('b', 'df')]
  assert.deepEqual(cellReads('b', ['df', 'model'], edges), ['df', 'model'])
})

test('the ambient client is not something a cell reads', () => {
  assert.deepEqual(cellReads('b', ['df', 'strata'], [edge('b', 'df')]), ['df'])
})

test('a cell that defines strata upstream does read it', () => {
  assert.deepEqual(cellReads('b', ['strata'], [edge('b', 'strata')]), ['strata'])
})

test('edges into other cells do not count', () => {
  assert.deepEqual(cellReads('b', ['df'], [edge('c', 'df')]), [])
})

test('an edge with no variable, such as an after ordering, adds no name', () => {
  assert.deepEqual(cellReads('b', ['strata'], [edge('b', '')]), [])
})
