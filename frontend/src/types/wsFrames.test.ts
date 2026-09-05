import assert from 'node:assert/strict'
import { test } from 'node:test'

import { isTypedFrame } from './notebook.ts'
import type { WsMessage } from './notebook.ts'

// Runtime behaviour only. This file is excluded from tsconfig.app.json and
// node --test strips types without checking them, so nothing here can pin a
// narrowing property -- the assertions in wsFrames.assertions.ts do that, and
// vue-tsc enforces them.

function frame(type: string, payload: unknown): WsMessage {
  return { type, seq: 1, ts: '2026-01-01T00:00:00Z', payload } as WsMessage
}

test('accepts a frame of the matching type', () => {
  const msg = frame('error', { error: 'busy', code: 'ENVIRONMENT_BUSY' })

  assert.ok(isTypedFrame(msg, 'error'))
})

test('rejects a frame of a different type', () => {
  // The payload map is keyed by frame name, so a wrong key must not narrow —
  // otherwise a renamed frame would keep type-checking against the old shape.
  const msg = frame('cell_status', { cell_id: 'c1', status: 'ready' })

  assert.equal(isTypedFrame(msg, 'error'), false)
})

test('does not match on payload shape, only on frame type', () => {
  // An error-shaped payload arriving under another frame name must not pass:
  // the predicate is the frame contract, not a duck-type check, so a frame
  // renamed on the server stops narrowing instead of quietly still matching.
  const msg = frame('cell_error', { error: 'boom', code: 'cell_busy' })

  assert.equal(isTypedFrame(msg, 'error'), false)
})
