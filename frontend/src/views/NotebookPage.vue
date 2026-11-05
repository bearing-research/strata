<script setup lang="ts">
import { ChevronDown, ChevronUp, Diamond, Play, RotateCw } from '@lucide/vue'
import { computed, defineAsyncComponent, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useNotebook } from '../stores/notebook'
import { useRecentNotebooks } from '../stores/recentNotebooks'
import AddCellMenu from '../components/AddCellMenu.vue'
import CellEditor from '../components/CellEditor.vue'
import ExportMenu from '../components/ExportMenu.vue'
import KeyboardShortcutsModal from '../components/KeyboardShortcutsModal.vue'
import OverflowMenu from '../components/OverflowMenu.vue'
import ThemeToggle from '../components/ThemeToggle.vue'
import { useStrata } from '../composables/useStrata'
import { embedSnippet } from '../utils/embedSnippet'
import { editorChrome, editorQuery, isFramed } from '../utils/framed'
import { BASE_PATH } from '../utils/strataBase'
import { clearNotebookPerfMarks, markNotebookPerf, measureNotebookPerf } from '../utils/perf'
import {
  connectionStatus,
  dagDrawerStartsCollapsed,
  NARROW_HEADER_QUERY,
  overflowActions,
  settingsAttention,
  sidebarIsRail,
} from '../utils/notebookChrome'

const DagView = defineAsyncComponent(() => import('../components/DagView.vue'))
const NotesPanel = defineAsyncComponent(() => import('../components/NotesPanel.vue'))
const EnvironmentPanel = defineAsyncComponent(() => import('../components/EnvironmentPanel.vue'))
const MountsPanel = defineAsyncComponent(() => import('../components/MountsPanel.vue'))
const ConnectionsPanel = defineAsyncComponent(() => import('../components/ConnectionsPanel.vue'))
const SchemaPanel = defineAsyncComponent(() => import('../components/SchemaPanel.vue'))
const RuntimePanel = defineAsyncComponent(() => import('../components/RuntimePanel.vue'))
const RegistryPanel = defineAsyncComponent(() => import('../components/RegistryPanel.vue'))
const LineageModal = defineAsyncComponent(() => import('../components/LineageModal.vue'))
const WorkersPanel = defineAsyncComponent(() => import('../components/WorkersPanel.vue'))
const ProfilingPanel = defineAsyncComponent(() => import('../components/ProfilingPanel.vue'))
const ImpactPreview = defineAsyncComponent(() => import('../components/ImpactPreview.vue'))
const InspectSwapConfirm = defineAsyncComponent(
  () => import('../components/InspectSwapConfirm.vue'),
)
const ReadConfirm = defineAsyncComponent(() => import('../components/ReadConfirm.vue'))

const props = defineProps<{ sessionId: string }>()

const route = useRoute()
// Framed by a platform (`?framed=1`): it has the brand, the links and delete.
const chrome = computed(() => editorChrome(route.query))
const router = useRouter()
const { record, remove, findBySessionId } = useRecentNotebooks()

async function copyEmbedSnippet() {
  await navigator.clipboard.writeText(
    embedSnippet(window.location.origin, BASE_PATH, props.sessionId),
  )
  pushToast('Embed snippet copied to clipboard')
}

const {
  notebook,
  orderedCells,
  variantDisplayCellId,
  connected,
  connectError,
  sessionClosed,
  sessionGone,
  notebookPath,
  hasUnsentEdits,
  flushDirtyCells,
  environmentMutationActive,
  environmentError,
  workerDefinitionsEditable,
  workerModeKnown,
  serviceMode,
  openBySessionId,
  openNotebook,
  updateNotebookNameAction,
  deleteNotebookAction,
  addCell,
  removeCell,
  moveCell,
  duplicateCell,
  executeCellWebSocket,
  executeNotebookRunAllWebSocket,
  executeRerunWebSocket,
  executeNotebookRerunAllWebSocket,
  cleanupWebSocket,
  ensureWorkersLoaded,
  registryEnabled,
  workersPanelShown,
  toasts,
  dismissToast,
  pushToast,
  setFramed,
} = useNotebook()
// Before the socket opens: it posts each focused cell to the framing page.
setFramed(isFramed(route.query))

// 'execution' is the DAG/Notes/Profiling stack; 'registry' needs registryEnabled.
const bottomTab = ref<'execution' | 'registry'>('execution')
function selectBottomTab(tab: 'execution' | 'registry') {
  bottomTab.value = tab
  if (dagDrawerCollapsed.value) toggleDagDrawer()
}

const editingName = ref(false)
const nameInput = ref<HTMLInputElement | null>(null)

// One slot per variant group: the active member in switch mode, the
// display-selected member in sweep mode (where every member is active).
const renderedCells = computed(() => {
  const sweepDisplay = new Map<string, string>()
  for (const g of notebook.variantGroups) {
    if (g.mode === 'sweep') sweepDisplay.set(g.group, variantDisplayCellId(g))
  }
  return orderedCells.value.filter((cell) => {
    if (cell.variantActive === false) return false
    if (cell.variantGroup && sweepDisplay.has(cell.variantGroup)) {
      return cell.id === sweepDisplay.get(cell.variantGroup)
    }
    return true
  })
})
// Deliberately fewer than AddCellMenu offers: get to a first cell, not a choice.
const emptyStateKinds = [
  { language: 'python', label: 'Python', description: 'Code cell (the default)' },
  { language: 'sql', label: 'SQL', description: 'Query a connected database' },
  { language: 'prompt', label: 'Prompt', description: 'LLM-powered template' },
  { language: 'markdown', label: 'Markdown', description: 'Documentation / prose' },
]

// The empty state stays up for the whole `addCell` round-trip, so without a
// latch a double-click creates two cells.
const addingFirstCell = ref(false)
async function addFirstCell(language: string) {
  if (addingFirstCell.value) return
  addingFirstCell.value = true
  try {
    await addCell(undefined, language)
  } finally {
    addingFirstCell.value = false
  }
}

const nameDraft = ref('')
const loading = ref(true)
const renamingNotebook = ref(false)
const renameError = ref<string | null>(null)
const deletingNotebook = ref(false)
const deleteError = ref<string | null>(null)
const reconnectError = ref<string | null>(null)
const recoveryPath = ref<string | null>(null)
const sidebarWidth = ref(340)
const showShortcuts = ref(false)

// --- DAG bottom drawer ---------------------------------------------------
// Large DAGs don't fit a narrow sidebar. Height and collapsed state persist.
const DAG_DRAWER_HEIGHT_KEY = 'strata:dagDrawerHeight'
const DAG_DRAWER_COLLAPSED_KEY = 'strata:dagDrawerCollapsed'
const DAG_DOCKED_KEY = 'strata:dagDocked'
const DAG_DRAWER_DEFAULT_HEIGHT = 320
const DAG_DRAWER_MIN_HEIGHT = 120
const DAG_DRAWER_MAX_HEIGHT_FRACTION = 0.8

function readNumber(key: string, fallback: number): number {
  try {
    const raw = localStorage.getItem(key)
    const n = raw == null ? NaN : Number(raw)
    return Number.isFinite(n) ? n : fallback
  } catch {
    return fallback
  }
}

const dagDrawerHeight = ref(readNumber(DAG_DRAWER_HEIGHT_KEY, DAG_DRAWER_DEFAULT_HEIGHT))
const dagDrawerCollapsed = ref(
  (() => {
    try {
      return dagDrawerStartsCollapsed(localStorage.getItem(DAG_DRAWER_COLLAPSED_KEY))
    } catch {
      return dagDrawerStartsCollapsed(null)
    }
  })(),
)

const dagDocked = ref(
  (() => {
    try {
      return localStorage.getItem(DAG_DOCKED_KEY) === '1'
    } catch {
      return false
    }
  })(),
)
const dagSelected = ref<string | null>(null)

function toggleDagDock() {
  dagDocked.value = !dagDocked.value
  // Back in the drawer, the graph is only visible with the drawer open on its tab.
  if (!dagDocked.value) selectBottomTab('execution')
  try {
    localStorage.setItem(DAG_DOCKED_KEY, dagDocked.value ? '1' : '0')
  } catch {
    // Storage blocked: the choice lasts for this page.
  }
}

function clampDagDrawerHeight(h: number): number {
  const maxH = Math.round(window.innerHeight * DAG_DRAWER_MAX_HEIGHT_FRACTION)
  return Math.max(DAG_DRAWER_MIN_HEIGHT, Math.min(maxH, h))
}

let dagResizePointerId: number | null = null
let dagResizeStartY = 0
let dagResizeStartHeight = 0

function startDagDrawerResize(event: PointerEvent) {
  if (dagDrawerCollapsed.value) return
  dagResizePointerId = event.pointerId
  dagResizeStartY = event.clientY
  dagResizeStartHeight = dagDrawerHeight.value
  document.body.classList.add('resizing-sidebar')
  window.addEventListener('pointermove', handleDagDrawerResize)
  window.addEventListener('pointerup', stopDagDrawerResize)
}

function handleDagDrawerResize(event: PointerEvent) {
  if (dagResizePointerId == null) return
  // Dragging up grows the drawer.
  dagDrawerHeight.value = clampDagDrawerHeight(
    dagResizeStartHeight - (event.clientY - dagResizeStartY),
  )
}

function stopDagDrawerResize() {
  dagResizePointerId = null
  document.body.classList.remove('resizing-sidebar')
  window.removeEventListener('pointermove', handleDagDrawerResize)
  window.removeEventListener('pointerup', stopDagDrawerResize)
  try {
    localStorage.setItem(DAG_DRAWER_HEIGHT_KEY, String(dagDrawerHeight.value))
  } catch {
    /* localStorage unavailable */
  }
}

function toggleDagDrawer() {
  dagDrawerCollapsed.value = !dagDrawerCollapsed.value
  try {
    localStorage.setItem(DAG_DRAWER_COLLAPSED_KEY, dagDrawerCollapsed.value ? '1' : '0')
  } catch {
    /* localStorage unavailable */
  }
}
let skipNextSessionReconnect: string | null = null

function isServiceModeSessionRestriction(message: string | null | undefined): boolean {
  const normalized = (message || '').toLowerCase()
  return (
    normalized.includes('service mode') ||
    normalized.includes('session apis are only available in personal mode')
  )
}

const narrowHeaderMedia = window.matchMedia(NARROW_HEADER_QUERY)
const narrowHeader = ref(narrowHeaderMedia.matches)
function onNarrowHeaderChange(event: MediaQueryListEvent) {
  narrowHeader.value = event.matches
}
narrowHeaderMedia.addEventListener('change', onNarrowHeaderChange)

const overflow = computed(() => overflowActions(chrome.value, narrowHeader.value))
const connection = computed(() => connectionStatus(loading.value, connected.value))

const settingsOpen = ref(false)
const attention = computed(() =>
  settingsAttention({
    mounts: notebook.mounts.length,
    worker: notebook.worker,
    environmentBusy: environmentMutationActive.value,
    environmentFailed:
      Boolean(environmentError.value) ||
      notebook.environment.syncState === 'failed' ||
      notebook.rEnvironment.syncState === 'failed',
  }),
)

const sidebarRail = computed(() => sidebarIsRail(settingsOpen.value, attention.value))

const notebookModeLabel = computed(() => (serviceMode.value ? 'Service mode' : 'Personal mode'))
const notebookModeTitle = computed(() =>
  workerDefinitionsEditable.value
    ? 'Notebook owns its worker catalog and local write flows.'
    : 'Workers and execution policy are controlled by the server.',
)
const serviceReconnectBlocked = computed(() =>
  isServiceModeSessionRestriction(reconnectError.value),
)
const reconnectHeading = computed(() =>
  serviceReconnectBlocked.value
    ? 'Service mode does not restore live sessions by URL'
    : sessionClosed.value
      ? 'Notebook closed'
      : 'Reconnect unavailable',
)
const reconnectSummary = computed(() =>
  serviceReconnectBlocked.value
    ? 'This URL points to a live notebook session. In service mode, browsers reopen notebooks by path and start a fresh session instead of reattaching to an existing one.'
    : reconnectError.value || 'Failed to reconnect to notebook session',
)
const reconnectActionLabel = computed(() =>
  serviceReconnectBlocked.value ? 'Reopen From Path' : 'Reopen Notebook',
)
const deleteButtonDisabled = computed(
  () => deletingNotebook.value || loading.value || environmentMutationActive.value,
)
const routeNotebookPath = computed(() => {
  const raw = route.query.path
  return typeof raw === 'string' && raw.trim() ? raw.trim() : null
})

let resizePointerId: number | null = null
let resizeStartX = 0
let resizeStartWidth = 340

function clampSidebarWidth(width: number): number {
  return Math.min(560, Math.max(280, width))
}

function handleGlobalKeydown(e: KeyboardEvent) {
  const tag = (e.target as HTMLElement)?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.closest('.cm-editor'))
    return
  if (e.key === '?' && !e.ctrlKey && !e.metaKey) {
    e.preventDefault()
    showShortcuts.value = !showShortcuts.value
  }
  if (e.key === 'Escape') {
    showShortcuts.value = false
  }
}

// Leaving with typing the server has not taken (the 2 s flush window, or the
// server is down) would lose it, so send what can go and ask about the rest.
function warnUnsentEdits(e: BeforeUnloadEvent) {
  flushDirtyCells()
  if (!hasUnsentEdits()) return
  e.preventDefault()
  // Browsers before preventDefault() asked for a prompt wanted returnValue set.
  e.returnValue = true
}

onMounted(async () => {
  window.addEventListener('keydown', handleGlobalKeydown)
  window.addEventListener('beforeunload', warnUnsentEdits)
  markNotebookPerf('notebook_page_mount')
  measureNotebookPerf('create_route_ms', 'create_route_start', 'notebook_page_mount')
  measureNotebookPerf('open_route_ms', 'open_route_start', 'notebook_page_mount')
  await connectToSession(props.sessionId)
  // Or the header mode badge stays at its fail-closed default until a worker
  // panel opens.
  void ensureWorkersLoaded()
})

watch(
  () => props.sessionId,
  async (newId) => {
    if (skipNextSessionReconnect === newId) {
      skipNextSessionReconnect = null
      return
    }
    cleanupWebSocket()
    await connectToSession(newId)
  },
)

// The server closed the session: say why and offer the existing reopen-by-path flow.
watch(sessionClosed, (closed) => {
  // Deleting from this page navigates home on its own.
  if (!closed || deletingNotebook.value) return
  reconnectError.value = closed.message
  recoveryPath.value =
    closed.reason === 'deleted'
      ? null
      : routeNotebookPath.value || findBySessionId(props.sessionId)?.path || null
})

// The server no longer has this session (it restarted): reopen by path as a new
// session. Typing it never got is carried over and sent once the socket opens.
watch(sessionGone, (gone) => {
  if (!gone || deletingNotebook.value) return
  recoveryPath.value =
    notebookPath.value || routeNotebookPath.value || findBySessionId(props.sessionId)?.path || null
  if (recoveryPath.value) {
    void reopenNotebookFromRecent()
  } else {
    reconnectError.value = 'The server no longer has this notebook session.'
  }
})

async function connectToSession(sessionId: string) {
  loading.value = true
  renameError.value = null
  deleteError.value = null
  reconnectError.value = null
  recoveryPath.value = null
  clearNotebookPerfMarks(
    'session_click',
    'connect_start',
    'notebook_ready',
    'connect_total_ms',
    'create_total_ms',
    'open_total_ms',
    'session_total_ms',
  )
  markNotebookPerf('session_click')
  markNotebookPerf('connect_start')
  try {
    const data = await openBySessionId(sessionId)
    if (data && data.path) {
      record(data.name || notebook.name, data.path, data.session_id || sessionId)
    }
    await nextTick()
    markNotebookPerf('notebook_ready')
    measureNotebookPerf('notebook_mount_to_ready_ms', 'notebook_page_mount', 'notebook_ready')
    measureNotebookPerf('connect_total_ms', 'connect_start', 'notebook_ready')
    measureNotebookPerf('create_total_ms', 'create_click', 'notebook_ready')
    measureNotebookPerf('open_total_ms', 'open_click', 'notebook_ready')
    measureNotebookPerf('session_total_ms', 'session_click', 'notebook_ready')
  } catch (e: any) {
    const message = e?.message || 'Failed to reconnect to notebook session'
    const recoveryCandidate = routeNotebookPath.value || findBySessionId(sessionId)?.path || null
    if (isServiceModeSessionRestriction(message) && recoveryCandidate) {
      try {
        const data = await openNotebook(recoveryCandidate)
        record(data.name || notebook.name, recoveryCandidate, data.session_id)
        if (typeof data.session_id === 'string' && data.session_id !== props.sessionId) {
          skipNextSessionReconnect = data.session_id
          await router.replace({
            name: 'notebook',
            params: { sessionId: data.session_id },
            query: editorQuery(route.query, recoveryCandidate),
          })
        }
        return
      } catch (reopenError: any) {
        reconnectError.value = reopenError?.message || message
        recoveryPath.value = recoveryCandidate
        return
      }
    }

    const recentEntry = findBySessionId(sessionId)
    if (recentEntry) {
      reconnectError.value = message
      recoveryPath.value = recentEntry.path
      return
    }

    console.error('Failed to open session:', e)
    // Framed, home would navigate the parent's frame away from the notebook.
    if (!chrome.value.pageLinks) {
      reconnectError.value = message
      return
    }
    router.replace({ name: 'home' })
    return
  } finally {
    loading.value = false
  }
}

async function reopenNotebookFromRecent() {
  if (!recoveryPath.value) return

  const path = recoveryPath.value
  loading.value = true
  reconnectError.value = null

  try {
    const data = await openNotebook(path)
    record(data.name || notebook.name, path, data.session_id)
    if (typeof data.session_id === 'string' && data.session_id !== props.sessionId) {
      skipNextSessionReconnect = data.session_id
      await router.replace({
        name: 'notebook',
        params: { sessionId: data.session_id },
        query: editorQuery(route.query, path),
      })
    }
  } catch (e: any) {
    reconnectError.value = e?.message || 'Failed to reopen notebook'
    recoveryPath.value = path
  } finally {
    loading.value = false
  }
}

async function deleteNotebook() {
  const notebookName = notebook.name || 'Untitled Notebook'
  const confirmed = window.confirm(
    `Delete notebook "${notebookName}" and all local environment and artifact data? This cannot be undone.`,
  )
  if (!confirmed) return

  deletingNotebook.value = true
  deleteError.value = null

  try {
    const data = await deleteNotebookAction()
    const deletedPath =
      (typeof data?.path === 'string' && data.path) ||
      routeNotebookPath.value ||
      findBySessionId(props.sessionId)?.path ||
      null
    if (deletedPath) {
      remove(deletedPath)
    }
    await router.replace({ name: 'home' })
  } catch (e: any) {
    deleteError.value = e?.message || 'Failed to delete notebook'
  } finally {
    deletingNotebook.value = false
  }
}

function stopSidebarResize() {
  resizePointerId = null
  document.body.classList.remove('resizing-sidebar')
  window.removeEventListener('pointermove', handleSidebarResize)
  window.removeEventListener('pointerup', stopSidebarResize)
}

function handleSidebarResize(event: PointerEvent) {
  if (resizePointerId == null) return
  sidebarWidth.value = clampSidebarWidth(resizeStartWidth - (event.clientX - resizeStartX))
}

function startSidebarResize(event: PointerEvent) {
  if (window.innerWidth <= 980) return
  resizePointerId = event.pointerId
  resizeStartX = event.clientX
  resizeStartWidth = sidebarWidth.value
  document.body.classList.add('resizing-sidebar')
  window.addEventListener('pointermove', handleSidebarResize)
  window.addEventListener('pointerup', stopSidebarResize)
}

onUnmounted(() => {
  narrowHeaderMedia.removeEventListener('change', onNarrowHeaderChange)
  window.removeEventListener('keydown', handleGlobalKeydown)
  window.removeEventListener('beforeunload', warnUnsentEdits)
  stopSidebarResize()
  stopDagDrawerResize()
  cleanupWebSocket()
})

function startEditName() {
  nameDraft.value = notebook.name
  renameError.value = null
  editingName.value = true
  setTimeout(() => nameInput.value?.select(), 0)
}

function cancelEditName() {
  nameDraft.value = notebook.name
  editingName.value = false
}

async function commitEditName() {
  if (!editingName.value) return

  const nextName = nameDraft.value.trim()
  if (!nextName) {
    renameError.value = 'Notebook name cannot be empty'
    setTimeout(() => nameInput.value?.select(), 0)
    return
  }
  if (nextName === notebook.name) {
    editingName.value = false
    return
  }

  renamingNotebook.value = true
  renameError.value = null

  try {
    const data = await updateNotebookNameAction(nextName)
    const path = routeNotebookPath.value || findBySessionId(props.sessionId)?.path || null
    if (path) {
      record(data.name || nextName, path, props.sessionId)
    }
    nameDraft.value = data.name || nextName
    editingName.value = false
  } catch (e: any) {
    renameError.value = e?.message || 'Failed to rename notebook'
    nameDraft.value = notebook.name
    setTimeout(() => nameInput.value?.select(), 0)
  } finally {
    renamingNotebook.value = false
  }
}

async function runCell(cellId: string) {
  if (environmentMutationActive.value) return
  const cell = orderedCells.value.find((c) => c.id === cellId)
  if (!cell || !cell.source.trim()) return
  executeCellWebSocket(cellId)
}

async function rerunCell(cellId: string) {
  if (environmentMutationActive.value) return
  const cell = orderedCells.value.find((c) => c.id === cellId)
  if (!cell || !cell.source.trim()) return
  executeRerunWebSocket(cellId)
}

async function runAll() {
  await executeNotebookRunAllWebSocket()
}

async function rerunAll() {
  await executeNotebookRerunAllWebSocket()
}

function onExportSelect(option: {
  key: string
  format?: 'markdown' | 'html'
  appView?: boolean
  embed?: boolean
}) {
  if (option.embed) {
    void copyEmbedSnippet()
    return
  }
  if (!option.format) return
  const sid = (notebook as any).sessionId as string | undefined
  if (!sid) return
  const strata = useStrata()
  strata.downloadExport(sid, option.format, Boolean(option.appView))
}

function goHome() {
  router.push({ name: 'home' })
}
</script>

<template>
  <div class="app" data-testid="notebook-page">
    <InspectSwapConfirm />
    <ReadConfirm />
    <!-- Header -->
    <header
      class="header"
      :data-mode="workerModeKnown ? (serviceMode ? 'service' : 'personal') : undefined"
    >
      <div v-if="chrome.brand" class="header-left">
        <span class="logo" role="button" tabindex="0" title="Home" @click="goHome"
          ><Diamond fill="currentColor" /> strata</span
        >
        <span
          v-if="!editingName"
          class="notebook-name"
          :title="notebook.name"
          @dblclick="startEditName"
        >
          {{ notebook.name }}
        </span>
        <input
          v-else
          ref="nameInput"
          v-model="nameDraft"
          class="name-input"
          :disabled="renamingNotebook"
          @blur="commitEditName"
          @keydown.enter.prevent="commitEditName"
          @keydown.esc.prevent="cancelEditName"
        />
      </div>
      <div class="header-right">
        <router-link
          v-if="chrome.pageLinks && !overflow.includes('app')"
          :to="`/app/${sessionId}`"
          class="header-logs-link"
          data-testid="nav-app"
          title="Open as read-only app (widgets + outputs)"
        >
          App
        </router-link>
        <ThemeToggle />
        <span
          v-if="chrome.modeBadge && !loading && notebook.id && workerModeKnown && serviceMode"
          class="mode-badge"
          :class="{
            service: !workerDefinitionsEditable,
            personal: workerDefinitionsEditable,
          }"
          :title="notebookModeTitle"
        >
          {{ notebookModeLabel }}
        </span>
        <span
          class="connection"
          :class="connection.state"
          role="status"
          data-testid="notebook-connection"
          :aria-label="connection.label"
          :title="connection.label"
        ></span>
        <button
          class="btn"
          data-testid="notebook-run-all"
          :disabled="!connected || environmentMutationActive"
          @click="runAll"
        >
          <Play fill="currentColor" /> Run All
        </button>
        <button
          v-if="!overflow.includes('rerunAll')"
          class="btn btn-secondary"
          data-testid="notebook-rerun-all"
          title="Force re-execute every cell, bypassing cache"
          :disabled="!connected || environmentMutationActive"
          @click="rerunAll"
        >
          <RotateCw /> Rerun All
        </button>
        <AddCellMenu
          variant="header"
          data-testid="notebook-add-cell"
          :disabled="!connected"
          @select="(language) => addCell(undefined, language)"
        />
        <ExportMenu
          data-testid="notebook-export"
          :disabled="!notebook.id"
          @select="onExportSelect"
        />
        <button
          v-if="!overflow.includes('shortcuts')"
          class="btn btn-secondary btn-shortcuts"
          title="Keyboard shortcuts (?)"
          @click="showShortcuts = true"
        >
          ?
        </button>
        <OverflowMenu v-if="overflow.length">
          <button
            v-if="overflow.includes('rerunAll')"
            class="overflow-item"
            role="menuitem"
            data-testid="notebook-rerun-all"
            title="Force re-execute every cell, bypassing cache"
            :disabled="!connected || environmentMutationActive"
            @click="rerunAll"
          >
            <RotateCw /> Rerun All
          </button>
          <router-link
            v-if="overflow.includes('app')"
            :to="`/app/${sessionId}`"
            class="overflow-item"
            role="menuitem"
            data-testid="nav-app"
            title="Open as read-only app (widgets + outputs)"
          >
            App
          </router-link>
          <button
            v-if="overflow.includes('shortcuts')"
            class="overflow-item"
            role="menuitem"
            title="Keyboard shortcuts (?)"
            @click="showShortcuts = true"
          >
            Keyboard shortcuts
          </button>
          <router-link
            v-if="overflow.includes('artifacts')"
            :to="`/notebook/${sessionId}/artifacts`"
            class="overflow-item"
            role="menuitem"
            data-testid="nav-artifacts"
            title="What this notebook has stored"
          >
            Artifacts
          </router-link>
          <router-link
            v-if="overflow.includes('logs')"
            to="/logs"
            class="overflow-item"
            role="menuitem"
            data-testid="nav-logs"
            title="Server logs"
          >
            Logs
          </router-link>
          <button
            v-if="overflow.includes('delete')"
            class="overflow-item overflow-item-danger"
            role="menuitem"
            data-testid="notebook-delete"
            :disabled="deleteButtonDisabled"
            :title="
              environmentMutationActive
                ? 'Finish the environment update before deleting the notebook.'
                : ''
            "
            @click="deleteNotebook"
          >
            {{ deletingNotebook ? 'Deleting…' : 'Delete Notebook' }}
          </button>
        </OverflowMenu>
      </div>
    </header>

    <!-- Connection error banner -->
    <div v-if="connectError && !loading" class="error-banner">
      Server not reachable: {{ connectError }}
    </div>

    <div v-else-if="deleteError && !loading" class="error-banner">
      {{ deleteError }}
    </div>

    <div v-else-if="renameError && !loading" class="error-banner">
      {{ renameError }}
    </div>

    <div v-else-if="reconnectError && !loading" class="reconnect-state">
      <h2>{{ reconnectHeading }}</h2>
      <p class="reconnect-message">{{ reconnectSummary }}</p>
      <p v-if="recoveryPath" class="reconnect-detail">
        {{
          serviceReconnectBlocked
            ? 'Reopen the notebook from its last known path to start a fresh session on the shared server:'
            : 'Reopen the notebook from its last known path:'
        }}
        <code>{{ recoveryPath }}</code>
      </p>
      <div class="reconnect-actions">
        <button
          v-if="recoveryPath"
          class="btn"
          :disabled="loading"
          @click="reopenNotebookFromRecent"
        >
          {{ reconnectActionLabel }}
        </button>
        <button
          v-if="chrome.pageLinks"
          class="btn btn-secondary"
          :disabled="loading"
          @click="goHome"
        >
          Back Home
        </button>
      </div>
    </div>

    <!-- Main layout -->
    <div
      v-else-if="!loading"
      class="workspace"
      :style="{
        '--sidebar-width': `${sidebarWidth}px`,
        '--dag-drawer-height': dagDrawerCollapsed ? '32px' : `${dagDrawerHeight}px`,
      }"
    >
      <div class="main">
        <!-- Cells -->
        <div class="cells-panel" data-testid="notebook-cells-panel">
          <CellEditor
            v-for="cell in renderedCells"
            :key="cell.id"
            :cell="cell"
            @run="runCell"
            @rerun="rerunCell"
            @delete="removeCell"
            @add-below="addCell"
            @duplicate="duplicateCell"
            @move-up="(id) => moveCell(id, 'up')"
            @move-down="(id) => moveCell(id, 'down')"
          />
          <!-- Empty notebook -->
          <div v-if="orderedCells.length === 0" class="cells-empty">
            <p class="cells-empty-title">This notebook is empty.</p>
            <p class="cells-empty-hint">
              Add a cell to start. Every cell you run is cached by its source, its inputs, and the
              environment, so re-running unchanged work costs nothing.
            </p>
            <div class="cells-empty-kinds">
              <button
                v-for="kind in emptyStateKinds"
                :key="kind.language"
                type="button"
                class="btn btn-secondary"
                :disabled="!connected || addingFirstCell"
                :title="kind.description"
                @click="addFirstCell(kind.language)"
              >
                {{ kind.label }}
              </button>
            </div>
            <p class="cells-empty-hint">
              Or browse the
              <a
                href="https://bearingresearch.com/strata/latest/notebook/examples/"
                target="_blank"
                rel="noopener"
              >
                example notebooks
              </a>
              to see what a finished one looks like.
            </p>
          </div>

          <div class="add-cell-row">
            <AddCellMenu
              variant="inline"
              :disabled="!connected"
              @select="(language) => addCell(undefined, language)"
            />
          </div>
        </div>

        <aside v-if="dagDocked" class="dag-dock" data-testid="dag-dock">
          <DagView v-model:selected="dagSelected" docked @toggle-dock="toggleDagDock" />
        </aside>

        <div
          v-if="!sidebarRail"
          class="sidebar-resizer"
          role="separator"
          aria-label="Resize sidebar"
          aria-orientation="vertical"
          @pointerdown="startSidebarResize"
        ></div>

        <!-- Runtime / config panels (ProfilingPanel is in the DAG drawer) -->
        <aside class="sidebar" :class="{ 'sidebar-rail': sidebarRail }">
          <!-- Panels stay mounted in one list, so moving in or out of the group keeps their state. -->
          <button
            class="settings-toggle"
            data-testid="notebook-settings-toggle"
            :aria-expanded="settingsOpen"
            @click="settingsOpen = !settingsOpen"
          >
            Notebook settings
            <span class="toggle-icon"><ChevronUp v-if="settingsOpen" /><ChevronDown v-else /></span>
          </button>
          <div
            v-show="settingsOpen || attention.mounts"
            :class="attention.mounts ? 'panel-attention' : 'panel-grouped'"
          >
            <MountsPanel :attention="attention.mounts" />
          </div>
          <div v-show="settingsOpen" class="panel-grouped"><ConnectionsPanel /></div>
          <div v-show="settingsOpen" class="panel-grouped"><SchemaPanel /></div>
          <div
            v-if="workersPanelShown"
            v-show="settingsOpen || attention.worker"
            :class="attention.worker ? 'panel-attention' : 'panel-grouped'"
          >
            <WorkersPanel :attention="attention.worker" />
          </div>
          <div v-show="settingsOpen" class="panel-grouped"><RuntimePanel /></div>
          <div
            v-show="settingsOpen || attention.environment"
            :class="attention.environment ? 'panel-attention' : 'panel-grouped'"
          >
            <EnvironmentPanel :attention="attention.environment" />
          </div>
        </aside>
      </div>

      <!-- DAG bottom drawer: drag the top edge to resize, click the header to collapse -->
      <div
        v-if="!dagDrawerCollapsed"
        class="dag-drawer-resizer"
        role="separator"
        aria-label="Resize DAG drawer"
        aria-orientation="horizontal"
        @pointerdown="startDagDrawerResize"
      ></div>
      <section class="dag-drawer" :class="{ collapsed: dagDrawerCollapsed }">
        <header class="dag-drawer-header">
          <div class="dag-drawer-tabs">
            <button
              class="drawer-tab"
              :class="{ active: bottomTab === 'execution' }"
              @click="selectBottomTab('execution')"
            >
              Execution
            </button>
            <button
              v-if="registryEnabled"
              class="drawer-tab"
              :class="{ active: bottomTab === 'registry' }"
              @click="selectBottomTab('registry')"
            >
              Registry
            </button>
          </div>
          <span class="dag-drawer-hint" @click="toggleDagDrawer">
            <template v-if="dagDrawerCollapsed"><ChevronUp /> expand</template>
            <template v-else><ChevronDown /> collapse</template>
          </span>
        </header>
        <div v-if="!dagDrawerCollapsed" class="dag-drawer-body">
          <template v-if="bottomTab === 'execution'">
            <div v-if="!dagDocked" class="dag-drawer-graph">
              <DagView v-model:selected="dagSelected" @toggle-dock="toggleDagDock" />
            </div>
            <NotesPanel />
            <div class="dag-drawer-profiling" :class="{ 'graph-docked': dagDocked }">
              <ProfilingPanel />
            </div>
          </template>
          <RegistryPanel v-else-if="bottomTab === 'registry'" />
        </div>
      </section>
    </div>

    <!-- Impact preview dialog -->
    <ImpactPreview />
    <KeyboardShortcutsModal :visible="showShortcuts" @close="showShortcuts = false" />

    <!-- Lineage modal -->
    <LineageModal />

    <!-- Registry feedback toasts -->
    <div class="toast-stack">
      <div
        v-for="t in toasts"
        :key="t.id"
        class="toast"
        :class="t.kind"
        @click="dismissToast(t.id)"
      >
        {{ t.message }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.cells-empty {
  padding: 32px 24px;
  border: 1px dashed var(--border);
  border-radius: 8px;
  text-align: center;
}

.cells-empty-title {
  margin: 0 0 8px;
  font-weight: 600;
  color: var(--text-primary);
}

.cells-empty-hint {
  margin: 0 auto;
  max-width: 46em;
  font-size: 13px;
  line-height: 1.5;
  color: var(--text-secondary);
}

.cells-empty-kinds {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin: 16px 0;
}

.add-cell-row {
  display: flex;
  gap: 8px;
  margin-top: 4px;
}

.add-cell-row .add-cell-btn {
  flex: 1;
}

.add-prompt-btn {
  border-color: var(--tint-primary-strong);
  color: var(--accent-primary);
}

.add-prompt-btn:hover {
  border-color: var(--accent-primary);
}

/* Shared by the <a> links and the Embed <button>; the resets keep them identical. */
.header-logs-link {
  font-family: inherit;
  font-size: 12px;
  line-height: 1.5;
  color: var(--text-secondary);
  text-decoration: none;
  padding: 3px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: transparent;
  cursor: pointer;
}

.header-logs-link:hover {
  color: var(--text-primary);
  border-color: var(--border-strong);
}

.mode-badge {
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--text-primary);
  background: var(--bg-input);
}

.mode-badge.service {
  background: var(--tint-primary);
  color: var(--accent-primary);
}

.mode-badge.personal {
  background: var(--tint-success);
  color: var(--accent-success);
}

.sidebar {
  display: flex;
  flex-direction: column;
}

.panel-attention {
  order: 0;
}

.settings-toggle {
  order: 1;
  display: flex;
  align-items: center;
  width: 100%;
  margin-top: 12px;
  padding: 4px 0;
  background: none;
  border: none;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  cursor: pointer;
}

.settings-toggle:hover {
  color: var(--text-primary);
}

.settings-toggle .toggle-icon {
  margin-left: auto;
  font-size: 10px;
}

.panel-grouped {
  order: 2;
}

/* A phone can't hold both on one row: the name above, what acts on it below. */
@media (max-width: 600px) {
  .header {
    flex-wrap: wrap;
    gap: 8px;
    padding: 8px 12px;
  }

  .header-left {
    flex: 1 1 0;
  }

  .header-right {
    display: contents;
  }

  .header > .header-right > * {
    order: 2;
  }

  .header > .header-right > .theme-toggle,
  .header > .header-right > .connection {
    order: 0;
  }

  .header:has(.header-left)::after {
    content: '';
    order: 1;
    flex-basis: 100%;
  }

  /* Hung from the header, not their buttons, so a menu wider than the gap beside one stays on. */
  .header {
    position: relative;
  }

  .header :deep(.add-cell-menu),
  .header :deep(.export-menu),
  .header :deep(.overflow-menu) {
    position: static;
  }

  .header :deep(.add-cell-dropdown),
  .header :deep(.export-dropdown),
  .header :deep(.overflow-dropdown) {
    left: 12px;
    right: 12px;
    width: auto;
    min-width: 0;
    white-space: normal;
  }
}

@media (min-width: 981px) {
  .sidebar-rail {
    width: 40px;
    min-width: 40px;
    padding: 16px 8px;
    border-left: 1px solid var(--border-subtle);
    overflow: hidden;
  }

  .sidebar-rail .settings-toggle {
    margin-top: 0;
    gap: 8px;
    white-space: nowrap;
    writing-mode: vertical-rl;
  }

  .sidebar-rail .settings-toggle .toggle-icon {
    display: none;
  }
}

.reconnect-state {
  max-width: 720px;
  margin: 48px auto;
  padding: 24px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 12px;
}

.reconnect-state h2 {
  margin-bottom: 12px;
  font-size: 20px;
  color: var(--text-primary);
}

.reconnect-message {
  color: var(--accent-warning);
  margin-bottom: 12px;
  line-height: 1.5;
}

.reconnect-detail {
  color: var(--text-secondary);
  margin-bottom: 16px;
  line-height: 1.5;
}

.reconnect-detail code {
  color: var(--text-primary);
}

.reconnect-actions {
  display: flex;
  gap: 12px;
}

/* Flash highlight when jumping to a cell from the DAG view */
:deep(.dag-jump-highlight) {
  animation: dag-jump-flash 1.5s ease-out;
}
@keyframes dag-jump-flash {
  0% {
    box-shadow: 0 0 0 3px var(--tint-primary-strong);
  }
  100% {
    box-shadow: none;
  }
}

/* Bottom-drawer tabs (Execution | Registry) */
.dag-drawer-tabs {
  display: flex;
  gap: 2px;
}
.drawer-tab {
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  cursor: pointer;
}
.drawer-tab:hover {
  color: var(--text-primary);
}
.drawer-tab.active {
  color: var(--text-primary);
  border-bottom-color: var(--accent-primary, #3b82f6);
}

/* Registry feedback toasts */
.toast-stack {
  position: fixed;
  bottom: 16px;
  right: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 2000;
}
.toast {
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  min-width: 220px;
  max-width: 360px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18);
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
}
.toast.success {
  background: var(--tint-success, #e6f4ea);
  color: var(--accent-success, #1e7e34);
}
.toast.error {
  background: var(--tint-danger, #fdecea);
  color: var(--accent-danger, #c0392b);
}
.toast.info {
  background: var(--tint-primary, #e8f0fe);
  color: var(--accent-primary, #1a56db);
}
</style>
