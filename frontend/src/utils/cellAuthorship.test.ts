import assert from 'node:assert/strict'
import test from 'node:test'

import { authorBadgeLabel, authorTitle } from './cellAuthorship.ts'

test('a cell you wrote yourself gets no badge', () => {
  assert.equal(authorBadgeLabel('local', 'local'), null)
})

test('a cell with no recorded author gets no badge', () => {
  assert.equal(authorBadgeLabel(null, null), null)
  assert.equal(authorBadgeLabel(undefined, undefined), null)
})

test('an agent-written cell names the agent', () => {
  assert.equal(authorBadgeLabel('agent:claude', 'agent:claude'), 'by agent:claude')
})

test('the last editor wins when someone other than you edited it', () => {
  assert.equal(authorBadgeLabel('local', 'assistant'), 'by assistant')
})

test('an agent cell you have since edited still says who started it', () => {
  assert.equal(authorBadgeLabel('agent:claude', 'local'), 'added by agent:claude')
})

test('in service mode every author is a principal, so every cell is badged', () => {
  assert.equal(authorBadgeLabel('alice', 'bob'), 'by bob')
})

test('the tooltip gives both, and says when one was not recorded', () => {
  assert.equal(authorTitle('agent:claude', 'local'), 'Added by agent:claude · last edited by local')
  assert.equal(authorTitle(null, 'local'), 'Added by unrecorded · last edited by local')
})
