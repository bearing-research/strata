import assert from 'node:assert/strict'
import { test } from 'node:test'

import { isTypedFrame } from './notebook.ts'
import type { WsMessage } from './notebook.ts'

// Runtime behaviour only; type narrowing is pinned in wsFrames.assertions.ts.

function frame(type: string, payload: unknown): WsMessage {
  return { type, seq: 1, ts: '2026-01-01T00:00:00Z', payload } as WsMessage
}

test('accepts a frame of the matching type', () => {
  const msg = frame('error', { error: 'busy', code: 'ENVIRONMENT_BUSY' })

  assert.ok(isTypedFrame(msg, 'error'))
})

test('rejects a frame of a different type', () => {
  // A wrong key must not narrow, or a renamed frame keeps the old shape.
  const msg = frame('cell_status', { cell_id: 'c1', status: 'ready' })

  assert.equal(isTypedFrame(msg, 'error'), false)
})

test('does not match on payload shape, only on frame type', () => {
  // The predicate checks the frame name, not the payload's shape.
  const msg = frame('cell_error', { error: 'boom', code: 'cell_busy' })

  assert.equal(isTypedFrame(msg, 'error'), false)
})
