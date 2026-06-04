import assert from 'node:assert/strict'
import test from 'node:test'

import { isStructuredStream, parsePartialJson } from './partialJson.ts'

test('complete JSON parses as-is', () => {
  assert.deepEqual(parsePartialJson('{"a": 1, "b": [true, null]}'), {
    a: 1,
    b: [true, null],
  })
})

test('non-JSON text returns undefined', () => {
  assert.equal(parsePartialJson('Once upon a time'), undefined)
  assert.equal(parsePartialJson(''), undefined)
})

test('open object and array are closed', () => {
  assert.deepEqual(parsePartialJson('{"items": [{"id": 1}, {"id": 2}'), {
    items: [{ id: 1 }, { id: 2 }],
  })
})

test('truncated mid-string closes the string', () => {
  assert.deepEqual(parsePartialJson('{"one_liner": "Costs ate the ed'), {
    one_liner: 'Costs ate the ed',
  })
})

test('dangling key with colon becomes null', () => {
  assert.deepEqual(parsePartialJson('{"a": 1, "b":'), { a: 1, b: null })
})

test('trailing comma is dropped', () => {
  assert.deepEqual(parsePartialJson('{"a": 1,'), { a: 1 })
  assert.deepEqual(parsePartialJson('[1, 2,'), [1, 2])
})

test('partial literal tail is trimmed back to the last separator', () => {
  assert.deepEqual(parsePartialJson('{"a": "x", "flag": tru'), {
    a: 'x',
    flag: null,
  })
  assert.deepEqual(parsePartialJson('[1, 2, 3.'), [1, 2])
})

test('half-typed key without colon falls back to undefined', () => {
  // Nothing structurally complete to show yet — the caller renders the
  // char ticker instead.
  assert.equal(parsePartialJson('{"on'), undefined)
})

test('escaped quotes inside strings do not end the string', () => {
  assert.deepEqual(parsePartialJson('{"quote": "she said \\"hi'), {
    quote: 'she said "hi',
  })
})

test('braces inside strings are not treated as structure', () => {
  assert.deepEqual(parsePartialJson('{"code": "if (x) { return [1"'), {
    code: 'if (x) { return [1',
  })
})

test('nested realistic schema output grows field by field', () => {
  // Simulates the review_triage stream at three cut points.
  const full =
    '{"items": [{"review_index": 0, "sentiment": "negative", ' +
    '"priority": "high", "tags": ["refund"]}, {"review_index": 1, "sen'

  const early = parsePartialJson(full.slice(0, 30))
  assert.ok(early !== undefined)

  const mid = parsePartialJson(full.slice(0, 95)) as { items: unknown[] }
  assert.equal(mid.items.length, 1)

  const late = parsePartialJson(full) as { items: Array<Record<string, unknown>> }
  assert.equal(late.items[0].priority, 'high')
  assert.equal(late.items.length, 2)
})

test('isStructuredStream sniffs objects and arrays only', () => {
  assert.equal(isStructuredStream('  {"a": 1'), true)
  assert.equal(isStructuredStream('[1, 2'), true)
  assert.equal(isStructuredStream('plain prose'), false)
  assert.equal(isStructuredStream(''), false)
})
