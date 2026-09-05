import assert from 'node:assert/strict'
import { test } from 'node:test'

import { isTypedFrame } from './notebook.ts'
import type { WsMessage } from './notebook.ts'

function frame(type: string, payload: unknown): WsMessage {
  return { type, seq: 1, ts: '2026-01-01T00:00:00Z', payload } as WsMessage
}

test('narrows a matching frame to its generated payload', () => {
  const msg = frame('error', { error: 'busy', code: 'ENVIRONMENT_BUSY' })

  assert.ok(isTypedFrame(msg, 'error'))
  if (isTypedFrame(msg, 'error')) {
    // The point of the helper: `code` is reachable without a cast, and a
    // mistyped code would be a compile error rather than a dead branch.
    assert.equal(msg.payload.code, 'ENVIRONMENT_BUSY')
    assert.equal(msg.payload.error, 'busy')
  }
})

test('rejects a frame of a different type', () => {
  // The payload map is keyed by frame name, so a wrong key must not narrow —
  // otherwise a renamed frame would keep type-checking against the old shape.
  const msg = frame('cell_status', { cell_id: 'c1', status: 'ready' })

  assert.equal(isTypedFrame(msg, 'error'), false)
})

test('a frame with no payload model is not typed', () => {
  // notebook_state has no model yet; its payload stays `unknown` and the
  // helper must not claim otherwise.
  const msg = frame('notebook_state', { cells: [] })

  assert.equal(isTypedFrame(msg, 'error'), false)
})
