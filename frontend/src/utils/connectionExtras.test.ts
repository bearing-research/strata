import assert from 'node:assert/strict'
import test from 'node:test'

import { connectionExtras } from './connectionExtras.ts'

const KNOWN = new Set(['name', 'driver', 'path'])

test('the server defaults on a plain connection are not extras', () => {
  // What the server sends back for a SQLite connection saved from the form.
  const spec = { name: 'local', driver: 'sqlite', path: 'a.db', options: {}, credential: null }
  assert.deepEqual(connectionExtras(spec, KNOWN), {})
})

test('settings the form does not edit are kept', () => {
  const spec = {
    name: 'lake',
    driver: 'duckdb',
    path: ':memory:',
    catalog: 'lake',
    mounts: ['raw'],
    options: { threads: 4 },
    credential: 'warehouse',
    tags: [],
  }
  assert.deepEqual(connectionExtras(spec, KNOWN), {
    catalog: 'lake',
    mounts: ['raw'],
    options: { threads: 4 },
    credential: 'warehouse',
  })
})
