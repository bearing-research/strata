import assert from 'node:assert/strict'
import test, { mock } from 'node:test'

import { useWebSocket } from './useWebSocket.ts'

// Stands in for the browser WebSocket; the test drives each instance's events.
class FakeSocket {
  static all: FakeSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((e: unknown) => void) | null = null
  onerror: ((e: unknown) => void) | null = null
  onclose: (() => void) | null = null
  sent: string[] = []

  constructor() {
    FakeSocket.all.push(this)
  }

  send(data: string) {
    this.sent.push(data)
  }

  close() {
    this.onclose?.()
  }

  // A refused or unreachable handshake: the browser fires error, then close.
  fail() {
    this.onerror?.({})
    this.onclose?.()
  }

  sentTypes(): string[] {
    return this.sent.map((raw) => JSON.parse(raw).type)
  }
}

;(globalThis as { WebSocket?: unknown }).WebSocket = FakeSocket

test.beforeEach(() => {
  FakeSocket.all = []
  mock.timers.enable({ apis: ['setTimeout'] })
  for (const level of ['log', 'warn', 'error'] as const) mock.method(console, level, () => {})
  // Every unexpected close probes the session; no test reaches a real server.
  fakeFetch(null)
})

test.afterEach(() => {
  mock.timers.reset()
  mock.restoreAll()
})

test('a dropped socket keeps retrying with backoff until one connects', () => {
  const ws = useWebSocket('nb')
  ws.connect()
  FakeSocket.all[0]!.onopen?.()
  assert.equal(ws.state.value, 'connected')

  FakeSocket.all[0]!.close()
  assert.equal(ws.state.value, 'reconnecting')

  // Each failed attempt schedules the next one, with a longer wait.
  for (const [i, delay] of [1000, 2000, 4000, 8000].entries()) {
    mock.timers.tick(delay - 1)
    assert.equal(FakeSocket.all.length, i + 1, 'not before the backoff elapses')
    mock.timers.tick(1)
    assert.equal(FakeSocket.all.length, i + 2)
    FakeSocket.all[i + 1]!.fail()
    assert.equal(ws.state.value, 'reconnecting')
  }

  mock.timers.tick(16000)
  const last = FakeSocket.all.at(-1)!
  last.onopen?.()
  assert.equal(ws.state.value, 'connected')
  assert.deepEqual(last.sentTypes(), ['notebook_sync'])
})

test('the backoff stops growing at 30 seconds and never gives up', () => {
  const ws = useWebSocket('nb')
  ws.connect()
  for (let i = 0; i < 15; i++) {
    FakeSocket.all.at(-1)!.fail()
    mock.timers.tick(30000)
  }
  assert.equal(FakeSocket.all.length, 16)
  assert.equal(ws.state.value, 'connecting')
})

test('disconnect stops reconnecting, even with a retry already scheduled', () => {
  const ws = useWebSocket('nb')
  ws.connect()
  FakeSocket.all[0]!.onopen?.()
  FakeSocket.all[0]!.close()
  ws.disconnect()

  mock.timers.tick(120000)
  assert.equal(FakeSocket.all.length, 1)
  assert.equal(ws.state.value, 'disconnected')
})

test('send reports whether the frame went out', () => {
  const ws = useWebSocket('nb')
  assert.equal(ws.updateCellSource('c1', 'x = 1'), false)
  ws.connect()
  FakeSocket.all[0]!.onopen?.()
  assert.equal(ws.updateCellSource('c1', 'x = 1'), true)
})

test('open handlers run on every connect, before the sync request', () => {
  const ws = useWebSocket('nb')
  ws.onOpen(() => ws.updateCellSource('c1', 'x = 1'))
  ws.connect()
  FakeSocket.all[0]!.onopen?.()
  FakeSocket.all[0]!.close()
  mock.timers.tick(1000)
  FakeSocket.all[1]!.onopen?.()

  for (const socket of FakeSocket.all) {
    assert.deepEqual(socket.sentTypes(), ['cell_source_update', 'notebook_sync'])
  }
})

// The server's answer to the session probe; `null` stands for an unreachable server.
function fakeFetch(status: number | null) {
  const calls: string[] = []
  const fetch = async (url: string) => {
    calls.push(url)
    if (status === null) throw new TypeError('Failed to fetch')
    return { status } as Response
  }
  mock.method(globalThis, 'fetch', fetch)
  return calls
}

// Lets the probe's fetch settle; the timers are mocked, setImmediate is not.
const settle = () => new Promise((resolve) => setImmediate(resolve))

test('a refused reconnect to a session the server no longer has stops retrying', async () => {
  const calls = fakeFetch(null)
  const ws = useWebSocket('old-session')
  let gone = 0
  ws.onSessionGone(() => gone++)
  ws.connect()
  FakeSocket.all[0]!.onopen?.()

  // The server goes down: the probe cannot reach it either, so keep retrying.
  FakeSocket.all[0]!.close()
  await settle()
  assert.equal(ws.state.value, 'reconnecting')
  assert.equal(gone, 0)
  assert.match(calls[0]!, /\/v1\/notebooks\/old-session\/dag$/)

  // It is back without the session: the upgrade is refused and the probe says 404.
  fakeFetch(404)
  mock.timers.tick(1000)
  FakeSocket.all[1]!.fail()
  await settle()

  assert.equal(gone, 1)
  assert.equal(ws.state.value, 'disconnected')
  mock.timers.tick(120000)
  assert.equal(FakeSocket.all.length, 2, 'no retry after the session is gone')
})

test('a session the server still has keeps reconnecting', async () => {
  fakeFetch(200)
  const ws = useWebSocket('nb')
  let gone = 0
  ws.onSessionGone(() => gone++)
  ws.connect()
  FakeSocket.all[0]!.fail()
  await settle()

  assert.equal(gone, 0)
  assert.equal(ws.state.value, 'reconnecting')
  mock.timers.tick(1000)
  assert.equal(FakeSocket.all.length, 2)
})

test('a socket closed by the client does not probe the session', async () => {
  const calls = fakeFetch(404)
  const ws = useWebSocket('nb')
  ws.connect()
  FakeSocket.all[0]!.onopen?.()
  ws.disconnect()
  await settle()
  assert.deepEqual(calls, [])
})
