import assert from 'node:assert/strict'
import test from 'node:test'
import { numericColumns } from './tableColumns.ts'

test('a column of numbers is numeric, and text is not', () => {
  assert.deepEqual(
    numericColumns(
      [
        ['East', 49, 21.86],
        ['West', 46, 76.31],
      ],
      3,
    ),
    [false, true, true],
  )
})

test('a null does not take a column out of line', () => {
  assert.deepEqual(numericColumns([[1.5], [null], [3.25], [undefined]], 1), [true])
})

test('a value that is not finite is still a number', () => {
  assert.deepEqual(numericColumns([[1], [Number.NaN], [Number.POSITIVE_INFINITY]], 1), [true])
})

test('the names the server gives NaN and infinity stay in a number column', () => {
  assert.deepEqual(numericColumns([[1.5], ['NaN'], ['inf'], ['-inf'], [null]], 1), [true])
})

test('text that happens to read NaN is still text', () => {
  assert.deepEqual(numericColumns([['NaN'], ['inf']], 1), [false])
  assert.deepEqual(numericColumns([['NaN'], ['n/a'], [1]], 1), [false])
})

test('a column with nothing to go on stays with the text', () => {
  assert.deepEqual(numericColumns([[null], [null]], 1), [false])
  assert.deepEqual(numericColumns([], 2), [false, false])
})

test('a column mixing numbers and text stays with the text', () => {
  assert.deepEqual(numericColumns([[1], ['two']], 1), [false])
})

test('a boolean is not a number', () => {
  assert.deepEqual(numericColumns([[true], [false]], 1), [false])
})
