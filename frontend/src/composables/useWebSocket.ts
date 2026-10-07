/** Notebook WebSocket: connection lifecycle with backoff, sequencing, dispatch. */

import { ref, shallowRef } from 'vue'
import type { WsMessage, WsClientMessageType, WsServerMessageType } from '../types/notebook'
import { BASE_PATH, strataHttpBase, strataWsBase } from '../utils/strataBase.ts'

const STRATA_HTTP_URL = strataHttpBase(
  (import.meta as any).env?.VITE_STRATA_URL,
  typeof window !== 'undefined' ? window.location.origin : undefined,
  BASE_PATH,
)
const STRATA_WS_URL = strataWsBase(STRATA_HTTP_URL)

export type WsConnectionState =
  'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error'

interface MessageHandler {
  (msg: WsMessage): void
}

export function useWebSocket(notebookId: string, options: { role?: string } = {}) {
  const connection = shallowRef<WebSocket | null>(null)
  const state = ref<WsConnectionState>('disconnected')
  const error = ref<string | null>(null)
  const clientSeq = ref(0)
  const messageHandlers = new Map<WsServerMessageType, MessageHandler[]>()
  const openHandlers: Array<() => void> = []
  const goneHandlers: Array<() => void> = []
  const reconnectAttempts = ref(0)
  // Only disconnect() stops reconnecting. Not read from `state`: a failed
  // attempt fires onerror (state 'error') before onclose.
  let closedByUser = false

  let _connectResolve: (() => void) | null = null
  let _connectReject: ((err: Error) => void) | null = null

  function connect(): void {
    if (state.value === 'connected' || state.value === 'connecting') {
      return
    }

    closedByUser = false
    state.value = 'connecting'
    error.value = null

    // App-view (read-only) connections declare `?role=viewer` so the server
    // rejects mutation frames.
    const roleQuery = options.role ? `?role=${encodeURIComponent(options.role)}` : ''
    const wsUrl = `${STRATA_WS_URL}/v1/notebooks/ws/${notebookId}${roleQuery}`

    try {
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('[WebSocket] Connected to notebook:', notebookId)
        state.value = 'connected'
        error.value = null
        reconnectAttempts.value = 0
        connection.value = ws
        // Before the sync, so the state it brings back already has them.
        openHandlers.forEach((handler) => handler())
        requestSync()
        if (_connectResolve) {
          _connectResolve()
          _connectResolve = null
          _connectReject = null
        }
      }

      ws.onmessage = (event: MessageEvent) => {
        try {
          const msg = JSON.parse(event.data) as WsMessage
          handleMessage(msg)
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e)
        }
      }

      ws.onerror = (event: Event) => {
        console.error('[WebSocket] Error:', event)
        state.value = 'error'
        error.value = 'Connection error'
        if (_connectReject) {
          _connectReject(new Error('WebSocket connection error'))
          _connectResolve = null
          _connectReject = null
        }
      }

      ws.onclose = () => {
        console.log('[WebSocket] Disconnected')
        if (connection.value === ws) connection.value = null

        if (closedByUser) {
          state.value = 'disconnected'
        } else {
          scheduleReconnect()
          void stopIfSessionGone()
        }
      }

      connection.value = ws
    } catch (e) {
      console.error('[WebSocket] Failed to create connection:', e)
      state.value = 'error'
      error.value = String(e)
      scheduleReconnect()
    }
  }

  function scheduleReconnect(): void {
    state.value = 'reconnecting'
    reconnectAttempts.value++

    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, then every 30s until disconnect().
    const delay = Math.min(1000 * 2 ** (reconnectAttempts.value - 1), 30000)

    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.value})`)

    setTimeout(() => {
      if (!closedByUser) connect()
    }, delay)
  }

  // The browser hides why an upgrade failed, so a refused one (403: the server
  // restarted and no longer has this session) looks like the server being down.
  // Ask over REST instead; only a 404 means gone, anything else keeps retrying.
  async function sessionGone(): Promise<boolean> {
    try {
      const resp = await fetch(
        `${STRATA_HTTP_URL}/v1/notebooks/${encodeURIComponent(notebookId)}/dag`,
      )
      return resp.status === 404
    } catch {
      return false
    }
  }

  async function stopIfSessionGone(): Promise<void> {
    if (!(await sessionGone()) || closedByUser) return
    console.warn('[WebSocket] Session is gone on the server:', notebookId)
    disconnect()
    goneHandlers.forEach((handler) => handler())
  }

  function disconnect(): void {
    closedByUser = true
    if (connection.value) {
      connection.value.close()
      connection.value = null
    }
    state.value = 'disconnected'
  }

  /** Send one frame; false when it could not go out (not connected). */
  function send(type: WsClientMessageType, payload: Record<string, any> = {}): boolean {
    if (state.value !== 'connected' || !connection.value) {
      console.warn('[WebSocket] Not connected, dropping message:', type)
      return false
    }

    clientSeq.value++
    const msg: WsMessage = {
      type,
      seq: clientSeq.value,
      ts: new Date().toISOString(),
      payload,
    }

    try {
      connection.value.send(JSON.stringify(msg))
      return true
    } catch (e) {
      console.error('[WebSocket] Failed to send message:', e)
      return false
    }
  }

  /** Run `handler` each time the socket opens, reconnects included. */
  function onOpen(handler: () => void): void {
    openHandlers.push(handler)
  }

  /** Run `handler` once the server says this session no longer exists; retries stop. */
  function onSessionGone(handler: () => void): void {
    goneHandlers.push(handler)
  }

  /** Register a handler for a message type; a type can have several. */
  function onMessage(type: WsServerMessageType, handler: MessageHandler): void {
    if (!messageHandlers.has(type)) {
      messageHandlers.set(type, [])
    }
    messageHandlers.get(type)!.push(handler)
  }

  function handleMessage(msg: WsMessage): void {
    const type = msg.type as WsServerMessageType
    const handlers = messageHandlers.get(type)

    if (handlers) {
      handlers.forEach((handler) => {
        try {
          handler(msg)
        } catch (e) {
          console.error('[WebSocket] Handler error for', type, ':', e)
        }
      })
    } else {
      console.warn('[WebSocket] No handlers for message type:', type)
    }
  }

  function requestSync(): void {
    send('notebook_sync', {})
  }

  function executeCell(cellId: string): void {
    send('cell_execute', { cell_id: cellId })
  }

  /** Execute all runnable notebook cells in notebook order. */
  function executeNotebookRunAll(): void {
    send('notebook_run_all', {})
  }

  function executeCascade(cellId: string, planId: string): void {
    send('cell_execute_cascade', { cell_id: cellId, plan_id: planId })
  }

  /** Execute cell with stale inputs ("Run this only"). */
  function executeForce(cellId: string): void {
    send('cell_execute_force', { cell_id: cellId })
  }

  /** Re-execute a cell bypassing its cache; upstreams materialize normally. */
  function executeRerun(cellId: string): void {
    send('cell_execute_rerun', { cell_id: cellId })
  }

  /** Force re-execute every cell in the notebook with cache off. */
  function executeNotebookRerunAll(): void {
    send('notebook_rerun_all', {})
  }

  function cancelCell(cellId: string): void {
    send('cell_cancel', { cell_id: cellId })
  }

  function updateCellSource(cellId: string, source: string, force = false): boolean {
    return send(
      'cell_source_update',
      force ? { cell_id: cellId, source, force } : { cell_id: cellId, source },
    )
  }

  /** Report this client's current cell (or null) for presence. */
  function focusCell(cellId: string | null): void {
    send('cell_focus', { cell_id: cellId })
  }

  function inspectOpen(cellId: string): void {
    send('inspect_open', { cell_id: cellId })
  }

  function inspectEval(cellId: string, expr: string): void {
    send('inspect_eval', { cell_id: cellId, expr })
  }

  function inspectClose(cellId: string): void {
    send('inspect_close', { cell_id: cellId })
  }

  /** Persist a cell's unit-test source and run it (Python cells only). */
  function runCellTests(cellId: string, testSource: string): void {
    send('cell_run_tests', { cell_id: cellId, test_source: testSource })
  }

  function addDependency(pkg: string): void {
    send('dependency_add', { package: pkg })
  }

  function removeDependency(pkg: string): void {
    send('dependency_remove', { package: pkg })
  }

  function setVariantActive(group: string, name: string): void {
    send('variant_set_active', { group, name })
  }

  /** Set widget control values; the backend persists them and stales downstream. */
  function sendWidgetUpdate(cellId: string, values: Record<string, unknown>): void {
    send('widget_update', { cell_id: cellId, values })
  }

  /** Add a variant to a group; the backend names it and makes it active. */
  function addVariant(group: string): void {
    send('variant_add', { group })
  }

  function debounceSourceUpdate(cellId: string, source: string, delayMs: number = 500): () => void {
    let timeoutId: number

    return () => {
      clearTimeout(timeoutId)
      timeoutId = window.setTimeout(() => {
        updateCellSource(cellId, source)
      }, delayMs)
    }
  }

  /** Wait for 'connected'. Rejects after timeoutMs or on connection error. */
  function waitForConnection(timeoutMs: number = 5000): Promise<void> {
    if (state.value === 'connected') return Promise.resolve()
    return new Promise<void>((resolve, reject) => {
      _connectResolve = resolve
      _connectReject = reject
      const timer = setTimeout(() => {
        _connectResolve = null
        _connectReject = null
        reject(new Error(`WebSocket connection timed out after ${timeoutMs}ms`))
      }, timeoutMs)
      const origResolve = _connectResolve
      _connectResolve = () => {
        clearTimeout(timer)
        origResolve?.()
      }
      const origReject = _connectReject
      _connectReject = (err) => {
        clearTimeout(timer)
        origReject?.(err)
      }
    })
  }

  const cleanup = () => {
    disconnect()
  }

  return {
    // State
    state,
    error,
    clientSeq,
    connected: () => state.value === 'connected',

    // Lifecycle
    connect,
    disconnect,
    cleanup,
    waitForConnection,

    // Messaging
    send,
    onMessage,
    onOpen,
    onSessionGone,

    // High-level actions
    requestSync,
    executeCell,
    executeNotebookRunAll,
    executeCascade,
    executeForce,
    executeRerun,
    executeNotebookRerunAll,
    cancelCell,
    updateCellSource,
    focusCell,
    debounceSourceUpdate,
    inspectOpen,
    inspectEval,
    inspectClose,
    runCellTests,
    addDependency,
    removeDependency,
    setVariantActive,
    addVariant,
    sendWidgetUpdate,
  }
}
