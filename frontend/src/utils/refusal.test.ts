import assert from 'node:assert/strict'
import test from 'node:test'

import { refusalNotice } from './refusal.ts'

test("a read-only refusal shows the server's sentence", () => {
  const sentence =
    "This organization is read-only. ('notebook_run_all' is refused; reads still work.)"
  assert.equal(refusalNotice({ error: sentence, code: 'read_only' }), sentence)
})

test('a missing scope shows its sentence too', () => {
  assert.equal(
    refusalNotice({
      error: "'cell_execute' requires the notebook:execute scope",
      code: 'insufficient_scope',
    }),
    "'cell_execute' requires the notebook:execute scope",
  )
})

test('a refusal with no sentence still says why', () => {
  assert.match(refusalNotice({ code: 'read_only' }) ?? '', /read-only/)
  assert.match(refusalNotice({ code: 'insufficient_scope', error: '  ' }) ?? '', /permission/)
})

test('other errors are left to their own handlers', () => {
  assert.equal(refusalNotice({ code: 'ENVIRONMENT_BUSY', error: 'busy' }), null)
  assert.equal(refusalNotice({ code: 'cell_locked', error: 'locked' }), null)
  assert.equal(refusalNotice({ error: 'Unknown message type: x' }), null)
  assert.equal(refusalNotice(null), null)
  assert.equal(refusalNotice('read_only'), null)
})
