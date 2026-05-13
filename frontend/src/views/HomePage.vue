<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  useStrata,
  type DiscoveredNotebook,
  type ImportNotebookResponse,
} from '../composables/useStrata'
import { preloadNotebookRoute } from '../router'
import { useRecentNotebooks } from '../stores/recentNotebooks'
import { primePrefetchedNotebookSession } from '../utils/notebookSessionPrefetch'
import { clearNotebookPerfMarks, markNotebookPerf, measureNotebookPerf } from '../utils/perf'
import ImportReportModal from '../components/ImportReportModal.vue'
import ThemeToggle from '../components/ThemeToggle.vue'

const router = useRouter()
const strata = useStrata()
const { entries: recentNotebooks, record, remove } = useRecentNotebooks()

const FALLBACK_NOTEBOOK_PARENT_PATH = '/tmp/strata-notebooks'
const newName = ref('Untitled Notebook')
const newParentPath = ref(FALLBACK_NOTEBOOK_PARENT_PATH)
const availablePythonVersions = ref<string[]>([])
const selectedPythonVersion = ref('')
const pythonSelectionFixed = ref(true)
const showNewForm = ref(false)
const showOpenForm = ref(false)
const discoveredNotebooks = ref<DiscoveredNotebook[]>([])
const discoveryRoot = ref<string | null>(null)
const discoveryLoading = ref(false)
const discoveryError = ref<string | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const failedRecentPath = ref<string | null>(null)
const failedRecentName = ref<string | null>(null)

// ---- Jupyter import flow ----
const importInput = ref<HTMLInputElement | null>(null)
const importing = ref(false)
const importError = ref<string | null>(null)
const importResult = ref<ImportNotebookResponse | null>(null)
// Active when an ``.ipynb`` is being dragged over the page. The
// dragenter / dragleave events fire on every child element transit,
// so we count nested transitions and only clear when the counter
// hits zero, matching the standard "page-wide drop target" pattern.
const dragDepth = ref(0)
const isDragHovering = ref(false)

onMounted(async () => {
  try {
    const data = await strata.getNotebookRuntimeConfig()
    const defaultParentPath =
      typeof data?.default_parent_path === 'string' && data.default_parent_path.trim()
        ? data.default_parent_path
        : FALLBACK_NOTEBOOK_PARENT_PATH
    const configuredPythonVersions = Array.isArray(data?.available_python_versions)
      ? data.available_python_versions
          .map((value: unknown) => String(value || '').trim())
          .filter((value: string) => value.length > 0)
      : []
    availablePythonVersions.value = configuredPythonVersions
    selectedPythonVersion.value =
      typeof data?.default_python_version === 'string' && data.default_python_version.trim()
        ? data.default_python_version
        : configuredPythonVersions[0] || ''
    pythonSelectionFixed.value =
      data?.python_selection_fixed === true || configuredPythonVersions.length <= 1
    if (newParentPath.value === FALLBACK_NOTEBOOK_PARENT_PATH) {
      newParentPath.value = defaultParentPath
    }
  } catch (e) {
    console.warn('Failed to load notebook config, using fallback parent path', e)
  }
})

async function createNotebook() {
  if (!newName.value.trim()) return
  loading.value = true
  dismissError()
  void preloadNotebookRoute()
  clearNotebookPerfMarks('create_click', 'create_response', 'create_request_ms', 'create_total_ms')
  markNotebookPerf('create_click')
  try {
    const notebookPath = `${newParentPath.value.replace(/\/+$/, '')}/${newName.value}`
    const data = await strata.createNotebook(
      newParentPath.value,
      newName.value,
      selectedPythonVersion.value || null,
    )
    markNotebookPerf('create_response')
    measureNotebookPerf('create_request_ms', 'create_click', 'create_response')
    const resolvedPath = data.path || notebookPath
    primePrefetchedNotebookSession(data)
    record(data.name, resolvedPath, data.session_id)
    markNotebookPerf('create_route_start')
    await router.push({
      name: 'notebook',
      params: { sessionId: data.session_id },
      query: { path: resolvedPath },
    })
  } catch (e: any) {
    error.value = e.message || 'Failed to create notebook'
  } finally {
    loading.value = false
  }
}

function triggerImportPicker() {
  importInput.value?.click()
}

function onImportFileChange(event: Event) {
  const input = event.target as HTMLInputElement | null
  const file = input?.files?.[0]
  if (file) {
    void importNotebookFile(file)
  }
  // Reset so picking the same file again re-fires the change event.
  if (input) input.value = ''
}

async function importNotebookFile(file: File) {
  if (!file.name.toLowerCase().endsWith('.ipynb')) {
    importError.value = `Only .ipynb files are supported (got "${file.name}").`
    return
  }
  if (importing.value) {
    importError.value = 'An import is already in progress.'
    return
  }
  importing.value = true
  importError.value = null
  dismissError()
  void preloadNotebookRoute()
  try {
    const data = await strata.importNotebook(file)
    importResult.value = data
  } catch (e: any) {
    importError.value = e?.message || `Failed to import ${file.name}`
  } finally {
    importing.value = false
  }
}

async function openImportedNotebook() {
  const data = importResult.value
  if (!data) return
  const resolvedPath = data.path || ''
  primePrefetchedNotebookSession(data)
  if (resolvedPath) record(data.name, resolvedPath, data.session_id)
  await router.push({
    name: 'notebook',
    params: { sessionId: data.session_id },
    query: resolvedPath ? { path: resolvedPath } : {},
  })
  importResult.value = null
}

function dismissImportModal() {
  importResult.value = null
}

function dismissImportError() {
  importError.value = null
}

// ---- Page-wide drag-and-drop ----

function isIpynbDrag(event: DragEvent): boolean {
  const items = event.dataTransfer?.items
  if (!items || items.length === 0) return false
  // ``DataTransfer.items`` only exposes the *kind* during drag (not
  // the filename), so accept anything that looks like a file drop
  // and validate the extension on drop. Multi-file drops surface as
  // multiple items; we reject them on drop too.
  for (let i = 0; i < items.length; i++) {
    if (items[i].kind === 'file') return true
  }
  return false
}

function onWindowDragEnter(event: DragEvent) {
  if (!isIpynbDrag(event)) return
  event.preventDefault()
  dragDepth.value += 1
  isDragHovering.value = true
}

function onWindowDragOver(event: DragEvent) {
  if (!isIpynbDrag(event)) return
  // preventDefault on dragover is what tells the browser this is a
  // valid drop target; without it the drop event never fires.
  event.preventDefault()
}

function onWindowDragLeave(event: DragEvent) {
  if (!isIpynbDrag(event)) return
  dragDepth.value = Math.max(0, dragDepth.value - 1)
  if (dragDepth.value === 0) isDragHovering.value = false
}

function onWindowDrop(event: DragEvent) {
  if (!isIpynbDrag(event)) return
  event.preventDefault()
  dragDepth.value = 0
  isDragHovering.value = false
  const files = event.dataTransfer?.files
  if (!files || files.length === 0) return
  if (files.length > 1) {
    importError.value = `Drop one .ipynb at a time (got ${files.length}).`
    return
  }
  void importNotebookFile(files[0])
}

onMounted(() => {
  window.addEventListener('dragenter', onWindowDragEnter)
  window.addEventListener('dragover', onWindowDragOver)
  window.addEventListener('dragleave', onWindowDragLeave)
  window.addEventListener('drop', onWindowDrop)
})

onBeforeUnmount(() => {
  window.removeEventListener('dragenter', onWindowDragEnter)
  window.removeEventListener('dragover', onWindowDragOver)
  window.removeEventListener('dragleave', onWindowDragLeave)
  window.removeEventListener('drop', onWindowDrop)
})

async function loadDiscoveredNotebooks() {
  discoveryLoading.value = true
  discoveryError.value = null
  try {
    const data = await strata.discoverNotebooks()
    discoveredNotebooks.value = data.notebooks
    discoveryRoot.value = data.root
  } catch (e: any) {
    discoveryError.value = e.message || 'Failed to scan notebook directory'
    discoveredNotebooks.value = []
  } finally {
    discoveryLoading.value = false
  }
}

// Refresh the list whenever the "Open Existing" panel becomes visible
// so the user sees recent writes from other notebooks without a reload.
watch(showOpenForm, (visible) => {
  if (visible) void loadDiscoveredNotebooks()
})

async function openNotebook(path?: string) {
  const target = path
  if (!target) return
  loading.value = true
  dismissError()
  void preloadNotebookRoute()
  clearNotebookPerfMarks('open_click', 'open_response', 'open_request_ms', 'open_total_ms')
  markNotebookPerf('open_click')
  try {
    const data = await strata.openNotebook(target)
    markNotebookPerf('open_response')
    measureNotebookPerf('open_request_ms', 'open_click', 'open_response')
    const resolvedPath = data.path || target
    primePrefetchedNotebookSession(data)
    record(data.name, resolvedPath, data.session_id)
    markNotebookPerf('open_route_start')
    await router.push({
      name: 'notebook',
      params: { sessionId: data.session_id },
      query: { path: resolvedPath },
    })
  } catch (e: any) {
    error.value = e.message || 'Failed to open notebook'
    if (path) {
      const failedEntry = recentNotebooks.value.find((entry) => entry.path === path)
      failedRecentPath.value = path
      failedRecentName.value = failedEntry?.name ?? null
    }
  } finally {
    loading.value = false
  }
}

function dismissError() {
  error.value = null
  failedRecentPath.value = null
  failedRecentName.value = null
}

function forgetRecent(path: string) {
  // Local-only: drop from recents list without touching the directory.
  remove(path)
  if (failedRecentPath.value === path) {
    dismissError()
  }
}

async function deleteRecent(path: string, name: string) {
  // Destructive: rm -rf the notebook directory on disk + drop from recents.
  const confirmed = window.confirm(
    `Delete notebook "${name}"?\n\nThis permanently removes the directory:\n${path}\n\nThis cannot be undone.`,
  )
  if (!confirmed) return

  loading.value = true
  dismissError()
  try {
    await strata.deleteNotebookByPath(path)
  } catch {
    // If the backend returns 404, the directory is already gone —
    // fall through and remove from recents anyway.
  }
  remove(path)
  if (failedRecentPath.value === path) {
    dismissError()
  }
  loading.value = false
}

function formatTime(ts: number): string {
  const d = new Date(ts)
  const now = Date.now()
  const diff = now - ts
  if (diff < 60_000) return 'just now'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`
  return d.toLocaleDateString()
}
</script>

<template>
  <div class="home" data-testid="home-page">
    <div class="home-theme-slot">
      <ThemeToggle />
    </div>
    <div class="home-container">
      <div class="home-header">
        <span class="logo">◆ strata</span>
        <span class="subtitle">notebook</span>
      </div>

      <!-- Error banner -->
      <div v-if="error" class="error-banner">
        <div class="error-copy">
          <span>{{ error }}</span>
          <button
            v-if="failedRecentPath"
            type="button"
            class="btn-inline"
            data-testid="remove-failed-recent"
            @click="forgetRecent(failedRecentPath)"
          >
            Remove
            {{ failedRecentName ? `"${failedRecentName}"` : 'this notebook' }}
            from recents
          </button>
        </div>
        <button class="btn-dismiss" @click="dismissError">&times;</button>
      </div>

      <!-- Actions -->
      <div class="actions">
        <div
          class="action-card"
          data-testid="action-new-notebook"
          @click="showNewForm = true"
          @mouseenter="void preloadNotebookRoute()"
          @focusin="void preloadNotebookRoute()"
        >
          <div class="action-icon">+</div>
          <div class="action-label">New Notebook</div>
        </div>
        <div
          class="action-card"
          data-testid="action-open-notebook"
          @click="showOpenForm = true"
          @mouseenter="void preloadNotebookRoute()"
          @focusin="void preloadNotebookRoute()"
        >
          <div class="action-icon">📂</div>
          <div class="action-label">Open Existing</div>
        </div>
        <div
          class="action-card"
          data-testid="action-import-notebook"
          :class="{ 'action-card-disabled': importing }"
          :aria-disabled="importing || undefined"
          @click="!importing && triggerImportPicker()"
          @mouseenter="void preloadNotebookRoute()"
          @focusin="void preloadNotebookRoute()"
        >
          <div class="action-icon">📒</div>
          <div class="action-label">
            {{ importing ? 'Importing…' : 'Import from Jupyter' }}
          </div>
          <div class="action-hint" v-if="!importing">or drop a .ipynb anywhere</div>
        </div>
      </div>

      <input
        ref="importInput"
        type="file"
        accept=".ipynb,application/x-ipynb+json"
        style="display: none"
        data-testid="import-file-input"
        @change="onImportFileChange"
      />

      <div v-if="importError" class="import-error" role="alert" data-testid="import-error">
        {{ importError }}
        <button class="import-error-dismiss" type="button" @click="dismissImportError">×</button>
      </div>

      <!-- New notebook form -->
      <div v-if="showNewForm" class="form-card" data-testid="new-notebook-form">
        <h3>New Notebook</h3>
        <label class="form-label">
          Name
          <input
            v-model="newName"
            type="text"
            class="form-input"
            data-testid="new-notebook-name"
            placeholder="My Notebook"
            @keydown.enter="createNotebook"
          />
        </label>
        <label class="form-label">
          Parent directory
          <input
            v-model="newParentPath"
            type="text"
            class="form-input"
            data-testid="new-notebook-parent-path"
            :placeholder="FALLBACK_NOTEBOOK_PARENT_PATH"
          />
        </label>
        <label class="form-label">
          Python version
          <select
            v-model="selectedPythonVersion"
            class="form-input"
            data-testid="new-notebook-python-version"
            :disabled="pythonSelectionFixed || availablePythonVersions.length === 0"
          >
            <option v-for="version in availablePythonVersions" :key="version" :value="version">
              {{ version }}
            </option>
          </select>
          <span class="form-help">
            {{
              pythonSelectionFixed
                ? 'This deployment currently provides a fixed notebook Python version.'
                : 'Select the notebook-level Python version before creation.'
            }}
          </span>
        </label>
        <div class="form-actions">
          <button
            class="btn"
            data-testid="create-notebook-submit"
            :disabled="loading"
            @click="createNotebook"
          >
            Create
          </button>
          <button class="btn btn-secondary" @click="showNewForm = false">Cancel</button>
        </div>
      </div>

      <!-- Open notebook form -->
      <div v-if="showOpenForm" class="form-card" data-testid="open-notebook-form">
        <h3>Open Notebook</h3>
        <div class="discovery-root">
          Scanning <code>{{ discoveryRoot || '(storage root unknown)' }}</code>
          <button
            class="discovery-refresh"
            :disabled="discoveryLoading"
            title="Rescan for notebooks"
            @click="loadDiscoveredNotebooks"
          >
            ↻
          </button>
        </div>
        <div v-if="discoveryLoading" class="discovery-status">Scanning…</div>
        <div v-else-if="discoveryError" class="discovery-error">
          {{ discoveryError }}
        </div>
        <div v-else-if="!discoveredNotebooks.length" class="discovery-status">
          No notebooks found under the storage root.
        </div>
        <ul v-else class="discovery-list" data-testid="open-notebook-list">
          <li
            v-for="nb in discoveredNotebooks"
            :key="nb.path"
            class="discovery-item"
            :class="{ disabled: loading }"
            data-testid="open-notebook-item"
            :data-notebook-path="nb.path"
            @click="!loading && openNotebook(nb.path)"
          >
            <div class="discovery-name">{{ nb.name || nb.path.split('/').pop() }}</div>
            <div class="discovery-path">{{ nb.path }}</div>
          </li>
        </ul>
        <div class="form-actions">
          <button class="btn btn-secondary" @click="showOpenForm = false">Cancel</button>
        </div>
      </div>

      <!-- Recent notebooks -->
      <div v-if="recentNotebooks.length > 0" class="recent-section">
        <h3 class="section-title">Recent Notebooks</h3>
        <div class="recent-list">
          <div
            v-for="entry in recentNotebooks"
            :key="entry.path"
            class="recent-item"
            :data-testid="`recent-notebook-${entry.name}`"
            @click="openNotebook(entry.path)"
          >
            <div class="recent-info">
              <span class="recent-name">{{ entry.name }}</span>
              <span class="recent-path">{{ entry.path }}</span>
            </div>
            <div class="recent-meta">
              <span class="recent-time">{{ formatTime(entry.lastOpened) }}</span>
              <button
                type="button"
                class="recent-remove"
                :data-testid="`recent-delete-${entry.name}`"
                :aria-label="`Delete ${entry.name}`"
                title="Delete notebook directory from disk"
                @click.stop="deleteRecent(entry.path, entry.name)"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Loading overlay -->
      <div v-if="loading" class="loading-overlay" data-testid="home-loading">
        <div class="spinner"></div>
        <span>Loading notebook...</span>
      </div>
    </div>

    <!-- Drag-and-drop overlay for Jupyter import -->
    <div v-if="isDragHovering" class="drag-overlay" data-testid="drag-overlay" aria-hidden="true">
      <div class="drag-overlay-card">
        <div class="drag-overlay-icon">📒</div>
        <div class="drag-overlay-title">Drop to import</div>
        <div class="drag-overlay-hint">.ipynb files only, one at a time</div>
      </div>
    </div>

    <!-- Import-in-progress overlay -->
    <div v-if="importing" class="loading-overlay" data-testid="import-loading">
      <div class="spinner"></div>
      <span>Converting notebook and syncing environment…</span>
    </div>

    <!-- Post-import report modal -->
    <ImportReportModal
      v-if="importResult"
      :report="importResult.import_report"
      :notebook-name="importResult.name"
      @open="openImportedNotebook"
      @close="dismissImportModal"
    />
  </div>
</template>

<style scoped>
.home {
  display: flex;
  justify-content: center;
  padding: 80px 24px 40px;
  min-height: 100vh;
}

.home-container {
  width: 100%;
  max-width: 640px;
}

.home-header {
  text-align: center;
  margin-bottom: 48px;
}

.home-theme-slot {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 10;
}

.logo {
  font-weight: 700;
  font-size: 28px;
  color: var(--accent-primary);
  letter-spacing: -0.5px;
}

.subtitle {
  font-size: 28px;
  color: var(--text-muted);
  margin-left: 8px;
  font-weight: 300;
}

.error-banner {
  background: var(--tint-danger);
  border: 1px solid var(--accent-danger);
  border-radius: 8px;
  color: var(--accent-danger);
  padding: 10px 16px;
  font-size: 13px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}

.error-copy {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}

.btn-dismiss {
  background: none;
  border: none;
  color: var(--accent-danger);
  font-size: 18px;
  cursor: pointer;
  padding: 0 4px;
}

.btn-inline {
  align-self: flex-start;
  background: none;
  border: none;
  color: var(--accent-warning);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  padding: 0;
}

.btn-inline:hover {
  text-decoration: underline;
}

.actions {
  display: flex;
  gap: 16px;
  margin-bottom: 32px;
}

.action-card {
  flex: 1;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  text-align: center;
  cursor: pointer;
  transition:
    border-color 0.15s,
    background 0.15s;
}

.action-card:hover {
  border-color: var(--accent-primary);
  background: var(--bg-elevated);
}

.action-icon {
  font-size: 28px;
  margin-bottom: 8px;
}

.action-label {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.action-hint {
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 4px;
}

.action-card-disabled {
  pointer-events: none;
  opacity: 0.6;
}

/* ---- Jupyter import: drag overlay, error banner ---- */

.drag-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 999;
  pointer-events: none;
  /* Overlay is visual only; window-level drop handler catches the file. */
}

.drag-overlay-card {
  background: var(--bg-surface);
  border: 2px dashed var(--accent, #3b82f6);
  border-radius: 16px;
  padding: 2.5rem 3rem;
  text-align: center;
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.3);
}

.drag-overlay-icon {
  font-size: 56px;
  margin-bottom: 12px;
}

.drag-overlay-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.drag-overlay-hint {
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.import-error {
  margin: 12px 0;
  padding: 0.65rem 0.9rem;
  background: var(--bg-warning, #fef3c7);
  border: 1px solid var(--border-warning, #fde68a);
  color: var(--text-warning, #92400e);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.9rem;
}

.import-error-dismiss {
  background: transparent;
  border: 0;
  font-size: 1.2rem;
  line-height: 1;
  cursor: pointer;
  color: inherit;
  padding: 0 0.25rem;
}

.form-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
}

.form-card h3 {
  font-size: 16px;
  color: var(--text-primary);
  margin-bottom: 16px;
}

.form-label {
  display: block;
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 12px;
}

.form-input {
  display: block;
  width: 100%;
  padding: 8px 12px;
  margin-top: 4px;
  background: var(--bg-input);
  border: 1px solid var(--border-strong);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 14px;
}

.discovery-root {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 10px;
}
.discovery-root code {
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.discovery-refresh {
  margin-left: auto;
  background: transparent;
  border: 1px solid var(--border-strong);
  color: var(--text-secondary);
  border-radius: 4px;
  padding: 2px 8px;
  cursor: pointer;
  font-size: 12px;
}
.discovery-refresh:hover:not(:disabled) {
  background: var(--bg-hover);
  color: var(--text-primary);
}
.discovery-refresh:disabled {
  opacity: 0.4;
  cursor: default;
}
.discovery-status,
.discovery-error {
  font-size: 12px;
  color: var(--text-secondary);
  padding: 12px 4px;
}
.discovery-error {
  color: var(--accent-danger);
}
.discovery-list {
  list-style: none;
  padding: 0;
  margin: 0 0 12px 0;
  max-height: 320px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
}
.discovery-item {
  padding: 8px 12px;
  cursor: pointer;
  border-bottom: 1px solid var(--border);
}
.discovery-item:last-child {
  border-bottom: none;
}
.discovery-item:hover:not(.disabled) {
  background: var(--bg-hover);
}
.discovery-item.disabled {
  cursor: default;
  opacity: 0.5;
}
.discovery-name {
  color: var(--text-primary);
  font-size: 13px;
  font-weight: 500;
}
.discovery-path {
  color: var(--text-muted);
  font-size: 11px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  margin-top: 2px;
}

.form-input:focus {
  outline: none;
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 2px var(--ring-focus);
}

.form-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  margin-top: 16px;
}

.form-help {
  display: block;
  margin-top: 6px;
  color: var(--text-muted);
  font-size: 12px;
}

.section-title {
  font-size: 13px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 12px;
}

.recent-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.recent-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 16px;
  background: var(--bg-surface);
  border: 1px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  transition:
    border-color 0.15s,
    background 0.15s;
}

.recent-item:hover {
  border-color: var(--border);
  background: var(--bg-elevated);
}

.recent-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.recent-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.recent-path {
  font-size: 12px;
  color: var(--text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.recent-time {
  font-size: 12px;
  color: var(--text-muted);
  flex-shrink: 0;
}

.recent-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.recent-remove {
  background: none;
  border: 1px solid var(--border-strong);
  border-radius: 999px;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 12px;
  padding: 4px 10px;
  transition:
    border-color 0.15s,
    color 0.15s,
    background 0.15s;
}

.recent-remove:hover {
  background: var(--bg-elevated);
  border-color: var(--accent-danger);
  color: var(--accent-danger);
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: var(--overlay-scrim);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  z-index: 1000;
  color: var(--text-secondary);
  font-size: 14px;
}

.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid var(--border);
  border-top-color: var(--accent-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
