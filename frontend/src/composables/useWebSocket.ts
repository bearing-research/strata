/**
 * WebSocket connection management for real-time notebook execution updates.
 *
 * Handles:
 * - Connection lifecycle (connect, reconnect with backoff, disconnect)
 * - Message serialization/deserialization
 * - Sequence number tracking
 * - Event handlers for different message types
 */

import { ref, shallowRef } from 'vue'
import type { WsMessage, WsClientMessageType, WsServerMessageType } from '../types/notebook'

function resolveStrataWsBase(): string {
  const configured = (import.meta as any).env?.VITE_STRATA_URL
  const httpBase =
    configured || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8765')

  if (httpBase.startsWith('https://')) {
    return `wss://${httpBase.slice('https://'.length)}`
  }
  if (httpBase.startsWith('http://')) {
    return `ws://${httpBase.slice('http://'.length)}`
  }
  return httpBase.replace(/^http/, 'ws')
}

const STRATA_WS_URL = resolveStrataWsBase()

export type WsConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'error'

interface MessageHandler {
  (msg: WsMessage): void
}

export function useWebSocket(notebookId: string) {
  const connection = shallowRef<WebSocket | null>(null)
  const state = ref<WsConnectionState>('disconnected')
  const error = ref<string | null>(null)
  const clientSeq = ref(0)
  const messageHandlers = new Map<WsServerMessageType, MessageHandler[]>()
  const reconnectAttempts = ref(0)
  const maxReconnectAttempts = 10
  const reconnectDelay = ref(1000) // Start at 1s, backoff to 30s max

  // Pending connection promise resolvers
  let _connectResolve: (() => void) | null = null
  let _connectReject: ((err: Error) => void) | null = null

  /**
   * Connect to the WebSocket endpoint.
   */
  function connect(): void {
    if (state.value === 'connected' || state.value === 'connecting') {
      return
    }

    state.value = 'connecting'
    error.value = null

    const wsUrl = `${STRATA_WS_URL}/v1/notebooks/ws/${notebookId}`

    try {
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('[WebSocket] Connected to notebook:', notebookId)
        state.value = 'connected'
        error.value = null
        reconnectAttempts.value = 0
        reconnectDelay.value = 1000
        connection.value = ws
        requestSync()
        // Resolve any pending waitForConnection promise
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
        connection.value = null

        if (state.value === 'connected' || state.value === 'connecting') {
          // Unexpected close — try to reconnect
          scheduleReconnect()
        } else {
          state.value = 'disconnected'
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

  /**
   * Schedule a reconnection attempt with exponential backoff.
   */
  function scheduleReconnect(): void {
    if (reconnectAttempts.value >= maxReconnectAttempts) {
      state.value = 'error'
      error.value = 'Failed to reconnect after max attempts'
      return
    }

    state.value = 'reconnecting'
    reconnectAttempts.value++

    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s, 30s, ...
    const delay = Math.min(reconnectDelay.value * 2 ** (reconnectAttempts.value - 1), 30000)
    reconnectDelay.value = delay

    console.log(
      `[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.value}/${maxReconnectAttempts})`,
    )

    setTimeout(() => {
      connect()
    }, delay)
  }

  /**
   * Disconnect from the WebSocket.
   */
  function disconnect(): void {
    if (connection.value) {
      connection.value.close()
      connection.value = null
    }
    state.value = 'disconnected'
  }

  /**
   * Send a message to the server.
   */
  function send(type: WsClientMessageType, payload: Record<string, any> = {}): void {
    if (state.value !== 'connected' || !connection.value) {
      console.warn('[WebSocket] Not connected, dropping message:', type)
      return
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
    } catch (e) {
      console.error('[WebSocket] Failed to send message:', e)
    }
  }

  /**
   * Register a handler for a message type.
   * Multiple handlers can be registered for the same type.
   */
  function onMessage(type: WsServerMessageType, handler: MessageHandler): void {
    if (!messageHandlers.has(type)) {
      messageHandlers.set(type, [])
    }
    messageHandlers.get(type)!.push(handler)
  }

  /**
   * Handle incoming message — dispatch to registered handlers.
   */
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

  /**
   * Request notebook state (for reconnection or sync).
   */
  function requestSync(): void {
    send('notebook_sync', {})
  }

  /**
   * Execute a cell.
   */
  function executeCell(cellId: string): void {
    send('cell_execute', { cell_id: cellId })
  }

  /**
   * Execute all runnable notebook cells in notebook order.
   */
  function executeNotebookRunAll(): void {
    send('notebook_run_all', {})
  }

  /**
   * Execute cascade plan.
   */
  function executeCascade(cellId: string, planId: string): void {
    send('cell_execute_cascade', { cell_id: cellId, plan_id: planId })
  }

  /**
   * Execute cell with stale inputs ("Run this only").
   */
  function executeForce(cellId: string): void {
    send('cell_execute_force', { cell_id: cellId })
  }

  /**
   * Force re-execute a cell, bypassing its cache but still materializing
   * upstreams normally (stale upstreams refresh from the artifact store).
   */
  function executeRerun(cellId: string): void {
    send('cell_execute_rerun', { cell_id: cellId })
  }

  /**
   * Force re-execute every cell in the notebook with cache off.
   */
  function executeNotebookRerunAll(): void {
    send('notebook_rerun_all', {})
  }

  /**
   * Cancel a running cell.
   */
  function cancelCell(cellId: string): void {
    send('cell_cancel', { cell_id: cellId })
  }

  /**
   * Update cell source code.
   */
  function updateCellSource(cellId: string, source: string): void {
    send('cell_source_update', { cell_id: cellId, source })
  }

  /**
   * Open an inspect REPL for a cell.
   */
  function inspectOpen(cellId: string): void {
    send('inspect_open', { cell_id: cellId })
  }

  /**
   * Evaluate expression in an inspect REPL.
   */
  function inspectEval(cellId: string, expr: string): void {
    send('inspect_eval', { cell_id: cellId, expr })
  }

  /**
   * Close an inspect REPL.
   */
  function inspectClose(cellId: string): void {
    send('inspect_close', { cell_id: cellId })
  }

  /**
   * Add a package dependency.
   */
  function addDependency(pkg: string): void {
    send('dependency_add', { package: pkg })
  }

  /**
   * Remove a package dependency.
   */
  function removeDependency(pkg: string): void {
    send('dependency_remove', { package: pkg })
  }

  /**
   * Switch the active variant for a group.
   */
  function setVariantActive(group: string, name: string): void {
    send('variant_set_active', { group, name })
  }

  /**
   * Add a new variant to an existing group; the backend auto-names it
   * and switches it to active.
   */
  function addVariant(group: string): void {
    send('variant_add', { group })
  }

  /**
   * Debounced source update (for editor changes).
   */
  function debounceSourceUpdate(cellId: string, source: string, delayMs: number = 500): () => void {
    let timeoutId: number

    return () => {
      clearTimeout(timeoutId)
      timeoutId = window.setTimeout(() => {
        updateCellSource(cellId, source)
      }, delayMs)
    }
  }

  /**
   * Wait for the WebSocket to reach 'connected' state.
   * Resolves immediately if already connected.
   * Rejects after timeoutMs (default 5s) or on connection error.
   */
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
      // Clear the timeout if we resolve/reject before it fires
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

  // Auto-cleanup on component unmount
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
    debounceSourceUpdate,
    inspectOpen,
    inspectEval,
    inspectClose,
    addDependency,
    removeDependency,
    setVariantActive,
    addVariant,
  }
}
