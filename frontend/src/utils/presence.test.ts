import assert from 'node:assert/strict'
import test from 'node:test'

import { othersOnCell } from './presence.ts'

const entries = [
  { principal: 'alice', focused_cell_id: 'c1', since: 1 },
  { principal: 'bob', focused_cell_id: 'c1', since: 2 },
  { principal: 'carol', focused_cell_id: 'c2', since: 3 },
  { principal: 'dave', focused_cell_id: null, since: 4 },
]

test('a cell lists the others focused on it', () => {
  assert.deepEqual(othersOnCell(entries, 'c1', 'carol'), ['alice', 'bob'])
})

test('you are left out of your own cell', () => {
  assert.deepEqual(othersOnCell(entries, 'c1', 'alice'), ['bob'])
})

test('a cell nobody is on lists nobody', () => {
  assert.deepEqual(othersOnCell(entries, 'c3', null), [])
})
