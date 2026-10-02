import assert from 'node:assert/strict'
import test from 'node:test'

import { envDraftRows, envPayload, MASKED_ENV_VALUE } from './envMask.ts'

test('a masked value is shown blank and saved back as the marker', () => {
  const rows = envDraftRows({ API_KEY: MASKED_ENV_VALUE, LOG_LEVEL: 'info' })
  assert.deepEqual(rows, [
    { key: 'API_KEY', value: '', maskedKey: 'API_KEY' },
    { key: 'LOG_LEVEL', value: 'info', maskedKey: null },
  ])
  assert.deepEqual(envPayload(rows), { API_KEY: MASKED_ENV_VALUE, LOG_LEVEL: 'info' })
})

test('typing over a masked value replaces it', () => {
  const rows = envDraftRows({ API_KEY: MASKED_ENV_VALUE })
  rows[0].value = 'sk-new'
  assert.deepEqual(envPayload(rows), { API_KEY: 'sk-new' })
})

test('renaming a masked row does not carry the hidden value to the new name', () => {
  const rows = envDraftRows({ API_KEY: MASKED_ENV_VALUE })
  rows[0].key = 'OTHER_KEY'
  assert.deepEqual(envPayload(rows), { OTHER_KEY: '' })
})

test('blank keys are dropped and keys are trimmed', () => {
  const rows = envDraftRows({ DATABASE_URL: MASKED_ENV_VALUE })
  rows[0].key = ' DATABASE_URL '
  rows.push({ key: '  ', value: 'x', maskedKey: null })
  assert.deepEqual(envPayload(rows), { DATABASE_URL: MASKED_ENV_VALUE })
})
