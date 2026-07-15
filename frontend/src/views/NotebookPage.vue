<script setup lang="ts">
import { computed, defineAsyncComponent, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useNotebook } from '../stores/notebook'
import { useRecentNotebooks } from '../stores/recentNotebooks'
import AddCellMenu from '../components/AddCellMenu.vue'
import CellEditor from '../components/CellEditor.vue'
import ExportMenu from '../components/ExportMenu.vue'
import KeyboardShortcutsModal from '../components/KeyboardShortcutsModal.vue'
import ThemeToggle from '../components/ThemeToggle.vue'
import { useStrata } from '../composables/useStrata'
import { clearNotebookPerfMarks, markNotebookPerf, measureNotebookPerf } from '../utils/perf'

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
const LlmPanel = defineAsyncComponent(() => import('../components/LlmPanel.vue'))
const ImpactPreview = defineAsyncComponent(() => import('../components/ImpactPreview.vue'))
const InspectSwapConfirm = defineAsyncComponent(
  () => import('../components/InspectSwapConfirm.vue'),
)

const props = defineProps<{ sessionId: string }>()

const route = useRoute()
const router = useRouter()
const { record, remove, findBySessionId } = useRecentNotebooks()

// "Embed" — copy a ready-to-paste iframe snippet (app view in embed mode) plus
// the tiny auto-resize listener, so the notebook drops into another site.
const embedCopied = ref(false)
function embedSnippet(): string {
  const url = `${window.location.origin}/#/app/${props.sessionId}?embed=1`
  return [
    `<iframe src="${url}" title="Strata notebook" style="width:100%;border:0"></iframe>`,
    `<script>addEventListener('message',e=>{if(e.data&&e.data.type==='strata:embed:resize')`,
    `document.querySelector('iframe[title=\\'Strata notebook\\']').style.height=e.data.height+'px'})<\/script>`,
  ].join('\n')
}
async function copyEmbedSnippet() {
  await navigator.clipboard.writeText(embedSnippet())
  embedCopied.value = true
  setTimeout(() => (embedCopied.value = false), 1800)
}

const {
  notebook,
  orderedCells,
  variantDisplayCellId,
  connected,
  connectError,
  environmentMutationActive,
  workerDefinitionsEditable,
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
  toasts,
  dismissToast,
} = useNotebook()

// Bottom drawer tab: the existing DAG/Notes/Profiling stack ('execution')
// plus the new Registry dashboard ('registry', gated by registryEnabled).
const bottomTab = ref<'execution' | 'registry'>('execution')
function selectBottomTab(tab: 'execution' | 'registry') {
  bottomTab.value = tab
  if (dagDrawerCollapsed.value) toggleDagDrawer()
}

const editingName = ref(false)
const nameInput = ref<HTMLInputElement | null>(null)

// Variant cells render in a single slot per group. Switch mode: only the
// active member is `variantActive`, so the filter below keeps just it. Sweep
// mode: every member is active and runs, but the editor still shows one slot —
// the display-selected member (the tab strip swaps which), so collapse the rest.
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
// Large notebooks (10+ cells with many edges) don't fit well in a
// narrow right sidebar, so the DAG lives in a collapsible bottom
// drawer instead. Height + collapsed state are persisted to
// localStorage so they stick across page reloads.
const DAG_DRAWER_HEIGHT_KEY = 'strata:dagDrawerHeight'
const DAG_DRAWER_COLLAPSED_KEY = 'strata:dagDrawerCollapsed'
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
      return localStorage.getItem(DAG_DRAWER_COLLAPSED_KEY) === '1'
    } catch {
      return false
    }
  })(),
)

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
  // Dragging up grows the drawer (mouse Y decreases → height increases).
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

const notebookModeLabel = computed(() =>
  workerDefinitionsEditable.value ? 'Personal mode' : 'Service mode',
)
const notebookModeTitle = computed(() =>
  workerDefinitionsEditable.value
    ? 'Notebook owns its worker catalog and local write flows.'
    : 'Workers and execution policy are controlled by the shared server.',
)
const serviceReconnectBlocked = computed(() =>
  isServiceModeSessionRestriction(reconnectError.value),
)
const reconnectHeading = computed(() =>
  serviceReconnectBlocked.value
    ? 'Service mode does not restore live sessions by URL'
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

onMounted(async () => {
  window.addEventListener('keydown', handleGlobalKeydown)
  markNotebookPerf('notebook_page_mount')
  measureNotebookPerf('create_route_ms', 'create_route_start', 'notebook_page_mount')
  measureNotebookPerf('open_route_ms', 'open_route_start', 'notebook_page_mount')
  await connectToSession(props.sessionId)
  // Fetch the worker catalog on mount so the mode badge in the header
  // reflects the backend's actual deployment mode. Without this, the
  // badge stays at its fail-closed default (Service mode) until the
  // user opens a worker panel.
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
            query: { path: recoveryCandidate },
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
        query: { path },
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
  window.removeEventListener('keydown', handleGlobalKeydown)
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

function exportNotebook(format: 'markdown' | 'html', appView = false) {
  const sid = (notebook as any).sessionId as string | undefined
  if (!sid) return
  const strata = useStrata()
  strata.downloadExport(sid, format, appView)
}

function goHome() {
  router.push({ name: 'home' })
}
</script>

<template>
  <div class="app" data-testid="notebook-page">
    <InspectSwapConfirm />
    <!-- Header -->
    <header class="header">
      <div class="header-left">
        <span class="logo" role="button" tabindex="0" title="Home" @click="goHome">◆ strata</span>
        <span v-if="!editingName" class="notebook-name" @dblclick="startEditName">
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
          :to="`/app/${sessionId}`"
          class="header-logs-link"
          data-testid="nav-app"
          title="Open as read-only app (widgets + outputs)"
        >
          App
        </router-link>
        <button
          type="button"
          class="header-logs-link header-embed-btn"
          data-testid="nav-embed"
          :title="'Copy an <iframe> snippet to embed this app in another site'"
          @click="copyEmbedSnippet"
        >
          {{ embedCopied ? 'Copied ✓' : 'Embed' }}
        </button>
        <router-link
          to="/artifacts"
          class="header-logs-link"
          data-testid="nav-artifacts"
          title="Artifact store"
        >
          Artifacts
        </router-link>
        <router-link to="/logs" class="header-logs-link" data-testid="nav-logs" title="Server logs">
          Logs
        </router-link>
        <ThemeToggle />
        <span
          v-if="!loading && notebook.id"
          class="mode-badge"
          :class="{
            service: !workerDefinitionsEditable,
            personal: workerDefinitionsEditable,
          }"
          :title="notebookModeTitle"
        >
          {{ notebookModeLabel }}
        </span>
        <span class="connection" :class="{ connected: connected }">
          {{ loading ? '◌ Connecting…' : connected ? '● Live' : '○ Not connected' }}
        </span>
        <button
          class="btn"
          data-testid="notebook-run-all"
          :disabled="!connected || environmentMutationActive"
          @click="runAll"
        >
          ▶ Run All
        </button>
        <button
          class="btn"
          data-testid="notebook-rerun-all"
          title="Force re-execute every cell, bypassing cache"
          :disabled="!connected || environmentMutationActive"
          @click="rerunAll"
        >
          ↻ Rerun All
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
          @select="exportNotebook"
        />
        <button
          class="btn btn-danger"
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
        <button
          class="btn btn-secondary btn-shortcuts"
          title="Keyboard shortcuts (?)"
          @click="showShortcuts = true"
        >
          ?
        </button>
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
        <button class="btn" :disabled="loading" @click="reopenNotebookFromRecent">
          {{ reconnectActionLabel }}
        </button>
        <button class="btn btn-secondary" :disabled="loading" @click="goHome">Back Home</button>
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
          <div class="add-cell-row">
            <AddCellMenu
              variant="inline"
              :disabled="!connected"
              @select="(language) => addCell(undefined, language)"
            />
          </div>
        </div>

        <div
          class="sidebar-resizer"
          role="separator"
          aria-label="Resize sidebar"
          aria-orientation="vertical"
          @pointerdown="startSidebarResize"
        ></div>

        <!-- Runtime / config panels. ProfilingPanel lives in the
             DAG drawer alongside the graph — it reads per-cell
             execution stats that pair naturally with the DAG view. -->
        <aside class="sidebar">
          <MountsPanel />
          <ConnectionsPanel />
          <SchemaPanel />
          <WorkersPanel />
          <RuntimePanel />
          <EnvironmentPanel />
          <LlmPanel />
        </aside>
      </div>

      <!-- DAG bottom drawer. Drag its top edge to resize, click the
           header to collapse. Persisted to localStorage. -->
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
            {{ dagDrawerCollapsed ? '▲ expand' : '▼ collapse' }}
          </span>
        </header>
        <div v-if="!dagDrawerCollapsed" class="dag-drawer-body">
          <template v-if="bottomTab === 'execution'">
            <div class="dag-drawer-graph">
              <DagView />
            </div>
            <NotesPanel />
            <div class="dag-drawer-profiling">
              <ProfilingPanel />
            </div>
          </template>
          <RegistryPanel v-else-if="bottomTab === 'registry'" />
        </div>
      </section>
    </div>

    <!-- v1.1: Impact preview dialog -->
    <ImpactPreview />
    <KeyboardShortcutsModal :visible="showShortcuts" @close="showShortcuts = false" />

    <!-- Lineage modal (P3e) -->
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

.header-logs-link {
  font-size: 12px;
  color: var(--text-secondary);
  text-decoration: none;
  padding: 3px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
}

.header-logs-link:hover {
  color: var(--text-primary);
  border-color: var(--border-strong);
}

/* The Embed control is a <button> but should read like the sibling links. */
.header-embed-btn {
  background: none;
  font: inherit;
  line-height: normal;
  cursor: pointer;
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

.btn-danger {
  border-color: var(--accent-danger);
  background: var(--tint-danger);
  color: var(--accent-danger);
}

.btn-danger:disabled {
  opacity: 0.6;
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
  color: var(--text);
}
.drawer-tab.active {
  color: var(--text);
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
  background: var(--bg, #fff);
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
