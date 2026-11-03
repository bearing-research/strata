import assert from 'node:assert/strict'
import test from 'node:test'

import { editorChrome } from './framed.ts'
import {
  connectionStatus,
  dagDrawerStartsCollapsed,
  overflowActions,
  settingsAttention,
} from './notebookChrome.ts'

const standalone = editorChrome({})
const framed = editorChrome({ framed: '1' })

test('a wide top bar keeps its actions and overflows the links and delete', () => {
  assert.deepEqual(overflowActions(standalone, false), ['artifacts', 'logs', 'delete'])
})

test('a narrow top bar overflows its secondary actions too', () => {
  assert.deepEqual(overflowActions(standalone, true), [
    'rerunAll',
    'app',
    'shortcuts',
    'artifacts',
    'logs',
    'delete',
  ])
})

test('a framed editor offers no links and no delete in its menu', () => {
  assert.deepEqual(overflowActions(framed, false), [])
  assert.deepEqual(overflowActions(framed, true), ['rerunAll', 'shortcuts'])
})

test('the connection names its state, and loading wins', () => {
  assert.deepEqual(connectionStatus(false, true), { state: 'connected', label: 'Connected' })
  assert.deepEqual(connectionStatus(false, false), {
    state: 'disconnected',
    label: 'Not connected',
  })
  assert.deepEqual(connectionStatus(true, true), { state: 'connecting', label: 'Connecting' })
  assert.deepEqual(connectionStatus(true, false), { state: 'connecting', label: 'Connecting' })
})

test('the drawer opens collapsed with no saved choice, and a saved choice wins', () => {
  assert.equal(dagDrawerStartsCollapsed(null), true)
  assert.equal(dagDrawerStartsCollapsed('1'), true)
  assert.equal(dagDrawerStartsCollapsed('0'), false)
})

const quiet = { mounts: 0, worker: 'local', environmentBusy: false, environmentFailed: false }

test('a notebook with nothing to look at keeps every panel in the group', () => {
  assert.deepEqual(settingsAttention(quiet), { mounts: false, worker: false, environment: false })
  assert.equal(settingsAttention({ ...quiet, worker: null }).worker, false)
  assert.equal(settingsAttention({ ...quiet, worker: '' }).worker, false)
})

test('a mount, a remote worker or an environment in trouble leaves the group', () => {
  assert.equal(settingsAttention({ ...quiet, mounts: 1 }).mounts, true)
  assert.equal(settingsAttention({ ...quiet, worker: 'gpu-a' }).worker, true)
  assert.equal(settingsAttention({ ...quiet, environmentBusy: true }).environment, true)
  assert.equal(settingsAttention({ ...quiet, environmentFailed: true }).environment, true)
})
