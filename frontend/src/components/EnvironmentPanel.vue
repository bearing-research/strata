<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useNotebook } from '../stores/notebook'
import { useStrata } from '../composables/useStrata'
import PythonVersionModal from './PythonVersionModal.vue'

const {
  notebook,
  dependencies,
  resolvedDependencies,
  dependencyLoading,
  dependencyError,
  environmentLoading,
  environmentError,
  environmentWarnings,
  environmentLastAction,
  environmentOperation,
  environmentJobHistory,
  environmentMutationActive,
  environmentImportPreview,
  addDependencyAction,
  removeDependencyAction,
  syncEnvironmentAction,
  exportRequirementsAction,
  previewRequirementsImportAction,
  importRequirementsAction,
  previewEnvironmentYamlImportAction,
  importEnvironmentYamlAction,
  clearEnvironmentImportPreview,
  fetchDependencies,
  fetchEnvironment,
  connected,
  updatePythonVersionAction,
} = useNotebook()

const strata = useStrata()

// Python-version modal state. Versions are fetched lazily when the
// modal opens so the panel doesn't pay the round-trip cost on every
// notebook open — the picker is a rarely-used affordance.
const pythonModalOpen = ref(false)
const pythonModalBusy = ref(false)
const pythonModalError = ref<string | null>(null)
const pythonAvailable = ref<string[]>([])

async function openPythonModal() {
  pythonModalError.value = null
  if (pythonAvailable.value.length === 0) {
    try {
      const data = await strata.getNotebookRuntimeConfig()
      const raw = data as { available_python_versions?: unknown }
      pythonAvailable.value = Array.isArray(raw.available_python_versions)
        ? raw.available_python_versions
            .map((v: unknown) => String(v || '').trim())
            .filter((v: string) => v.length > 0)
        : []
    } catch (err) {
      pythonModalError.value = (err as Error).message || 'Failed to load available Python versions'
    }
  }
  pythonModalOpen.value = true
}

function closePythonModal() {
  if (pythonModalBusy.value) return
  pythonModalOpen.value = false
}

async function confirmPythonChange(version: string) {
  pythonModalBusy.value = true
  pythonModalError.value = null
  try {
    // The store action resolves the session id internally (the REST
    // route uses session_id, not notebook content id) and surfaces
    // server errors as ``error`` rather than throwing.
    const result = await updatePythonVersionAction(version)
    if (result.error) {
      pythonModalError.value = result.error
      return
    }
    // 200 no-op + 202 accepted both close the modal — the WS
    // environment_job_progress channel surfaces live progress, so
    // the modal doesn't have to stay open.
    pythonModalOpen.value = false
  } finally {
    pythonModalBusy.value = false
  }
}

const newPackage = ref('')
const showPanel = ref(false)
const packageView = ref<'declared' | 'resolved'>('declared')
const packageFilter = ref('')
const requirementsMode = ref<'requirements-import' | 'requirements-export' | 'yaml-import' | null>(
  null,
)
const requirementsText = ref('')
const importFileName = ref('')
const lastPreviewSignature = ref<string | null>(null)

const syncStateLabel = computed(() => {
  switch (notebook.environment.syncState) {
    case 'pending':
      return 'Initializing'
    case 'ready':
      return 'Ready'
    case 'fallback':
      return 'PATH Fallback'
    case 'failed':
      return 'Failed'
    default:
      return 'Unknown'
  }
})

const syncStateClass = computed(() => `state-${notebook.environment.syncState}`)
const syncButtonLabel = computed(() =>
  environmentMutationActive.value && environmentOperation.value?.action === 'sync'
    ? 'Rebuilding…'
    : 'Rebuild .venv',
)

const shortLockfileHash = computed(() =>
  notebook.environment.lockfileHash ? notebook.environment.lockfileHash.slice(0, 12) : 'none',
)

const shortRLockHash = computed(() =>
  notebook.rEnvironment.lockHash ? notebook.rEnvironment.lockHash.slice(0, 12) : 'none',
)

const rLastSyncedLabel = computed(() => {
  if (!notebook.rEnvironment.lastSyncedAt) return 'Not synced yet'
  return new Date(notebook.rEnvironment.lastSyncedAt).toLocaleString()
})

const lastSyncedLabel = computed(() => {
  if (!notebook.environment.lastSyncedAt) return 'Not synced yet'
  return new Date(notebook.environment.lastSyncedAt).toLocaleString()
})

const lastSyncDurationLabel = computed(() => {
  if (notebook.environment.lastSyncDurationMs == null) return 'Unknown'
  return `${notebook.environment.lastSyncDurationMs} ms`
})

const interpreterSourceLabel = computed(() => {
  switch (notebook.environment.interpreterSource) {
    case 'venv':
      return 'Notebook venv'
    case 'path':
      return 'PATH fallback'
    default:
      return 'Unknown'
  }
})

const filteredDeclaredDependencies = computed(() => {
  const query = packageFilter.value.trim().toLowerCase()
  if (!query) return dependencies.value
  return dependencies.value.filter((dep) =>
    `${dep.name} ${dep.specifier} ${dep.version}`.toLowerCase().includes(query),
  )
})

const filteredResolvedDependencies = computed(() => {
  const query = packageFilter.value.trim().toLowerCase()
  if (!query) return resolvedDependencies.value
  return resolvedDependencies.value.filter((dep) =>
    `${dep.name} ${dep.version}`.toLowerCase().includes(query),
  )
})

const currentPackageList = computed(() =>
  packageView.value === 'declared'
    ? filteredDeclaredDependencies.value
    : filteredResolvedDependencies.value,
)

const currentImportSignature = computed(() => {
  if (!requirementsMode.value || requirementsMode.value === 'requirements-export') return null
  return `${requirementsMode.value}:${requirementsText.value}`
})

const hasFreshImportPreview = computed(() => {
  if (!environmentImportPreview.value) return false
  if (lastPreviewSignature.value !== currentImportSignature.value) return false
  return (
    (requirementsMode.value === 'requirements-import' &&
      environmentImportPreview.value.kind === 'requirements') ||
    (requirementsMode.value === 'yaml-import' &&
      environmentImportPreview.value.kind === 'environment_yaml')
  )
})

const lastActionLabel = computed(() => {
  const action = environmentLastAction.value
  if (!action) return ''

  const verb =
    action.action === 'add'
      ? `Added ${action.packageName}`
      : action.action === 'remove'
        ? `Removed ${action.packageName}`
        : action.action === 'import'
          ? 'Imported environment file'
          : 'Rebuilt environment'
  const stale =
    action.staleCellCount > 0
      ? ` · ${action.staleCellCount} cell${action.staleCellCount === 1 ? '' : 's'} affected`
      : ''
  const lockfile = action.lockfileChanged ? ' · lockfile changed' : ' · lockfile unchanged'
  return `${verb}${lockfile}${stale}`
})

const operationTitle = computed(() => {
  if (!environmentOperation.value) return ''
  switch (environmentOperation.value.action) {
    case 'add':
      return 'Package install'
    case 'remove':
      return 'Package removal'
    case 'import':
      return 'Environment import'
    case 'sync':
      return 'Environment rebuild'
    default:
      return 'Environment operation'
  }
})

const operationStatusLabel = computed(() => {
  if (!environmentOperation.value) return ''
  switch (environmentOperation.value.status) {
    case 'running':
      return 'Running'
    case 'failed':
      return 'Failed'
    default:
      return 'Completed'
  }
})

const operationStatusClass = computed(() =>
  environmentOperation.value ? `operation-${environmentOperation.value.status}` : '',
)

const operationPhaseLabel = computed(() => {
  const phase = environmentOperation.value?.phase
  if (!phase) return ''
  return phase
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
})

const operationDurationLabel = computed(() => {
  if (!environmentOperation.value || environmentOperation.value.durationMs == null) return 'Timing…'
  return `${environmentOperation.value.durationMs} ms`
})

const recentEnvironmentOperations = computed(() =>
  environmentJobHistory.value
    .filter((operation) => operation.id !== environmentOperation.value?.id)
    .slice(0, 5),
)

function operationTimestampLabel(timestamp: number | null) {
  if (timestamp == null) return 'Unknown time'
  return new Date(timestamp).toLocaleString()
}

watch(
  showPanel,
  (open) => {
    if (open) {
      void fetchDependencies()
      void fetchEnvironment()
    }
  },
  { flush: 'post' },
)

async function addPackage() {
  const pkg = newPackage.value.trim()
  if (!pkg) return
  await addDependencyAction(pkg)
  newPackage.value = ''
}

async function removePackage(name: string) {
  await removeDependencyAction(name)
}

async function openRequirementsExport() {
  requirementsText.value = await exportRequirementsAction()
  importFileName.value = ''
  lastPreviewSignature.value = null
  clearEnvironmentImportPreview()
  requirementsMode.value = 'requirements-export'
}

function openRequirementsImport() {
  requirementsText.value = ''
  importFileName.value = ''
  lastPreviewSignature.value = null
  clearEnvironmentImportPreview()
  requirementsMode.value = 'requirements-import'
}

function openEnvironmentYamlImport() {
  requirementsText.value = ''
  importFileName.value = ''
  lastPreviewSignature.value = null
  clearEnvironmentImportPreview()
  requirementsMode.value = 'yaml-import'
}

function closeRequirementsEditor() {
  requirementsMode.value = null
  importFileName.value = ''
  lastPreviewSignature.value = null
  clearEnvironmentImportPreview()
}

async function previewImport() {
  if (requirementsMode.value === 'requirements-export') return
  if (requirementsMode.value === 'yaml-import') {
    await previewEnvironmentYamlImportAction(requirementsText.value)
  } else {
    await previewRequirementsImportAction(requirementsText.value)
  }
  if (!environmentError.value) {
    lastPreviewSignature.value = currentImportSignature.value
  }
}

async function applyRequirementsImport() {
  if (!hasFreshImportPreview.value) return
  if (requirementsMode.value === 'yaml-import') {
    await importEnvironmentYamlAction(requirementsText.value)
  } else {
    await importRequirementsAction(requirementsText.value)
  }
  if (!environmentError.value) {
    lastPreviewSignature.value = null
    requirementsMode.value = null
  }
}

async function handleImportFile(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  requirementsText.value = await file.text()
  importFileName.value = file.name
  lastPreviewSignature.value = null
  clearEnvironmentImportPreview()
  input.value = ''
}

function downloadRequirements() {
  const blob = new Blob([requirementsText.value], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'requirements.txt'
  link.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <div class="env-panel">
    <button class="env-toggle" @click="showPanel = !showPanel">
      Environment
      <span class="dep-count">{{ notebook.environment.declaredPackageCount }}</span>
      <span class="toggle-icon">{{ showPanel ? '&#9650;' : '&#9660;' }}</span>
    </button>

    <div v-if="showPanel" class="env-content">
      <div class="env-header">
        <div class="env-status-block">
          <div class="env-status">
            <span class="status-dot" :class="syncStateClass"></span>
            <span>{{ syncStateLabel }}</span>
          </div>
          <div class="env-status-help">
            Rebuilds the notebook <code>.venv</code> from <code>pyproject.toml</code> and
            <code>uv.lock</code>, refreshes runtime metadata, and resets warm execution state.
          </div>
        </div>
        <div class="env-actions">
          <button
            class="btn-sync"
            :disabled="
              !connected || environmentMutationActive || dependencyLoading || environmentLoading
            "
            @click="syncEnvironmentAction"
          >
            {{ syncButtonLabel }}
          </button>
          <button
            class="btn-secondary"
            :disabled="
              !connected || environmentMutationActive || dependencyLoading || environmentLoading
            "
            @click="openRequirementsExport"
          >
            Export
          </button>
          <button
            class="btn-secondary"
            :disabled="
              !connected || environmentMutationActive || dependencyLoading || environmentLoading
            "
            @click="openRequirementsImport"
          >
            Import requirements
          </button>
          <button
            class="btn-secondary"
            :disabled="
              !connected || environmentMutationActive || dependencyLoading || environmentLoading
            "
            @click="openEnvironmentYamlImport"
          >
            Import env.yaml
          </button>
        </div>
      </div>

      <div class="env-stats">
        <div class="env-stat">
          <span class="env-stat-label">Requested Python</span>
          <button
            type="button"
            class="env-stat-value env-stat-button"
            :disabled="environmentMutationActive"
            :title="
              environmentMutationActive
                ? 'Cannot change Python while another environment update is running'
                : 'Click to change the notebook\'s Python version'
            "
            @click="openPythonModal"
          >
            {{ notebook.environment.requestedPythonVersion || 'Unknown' }}
          </button>
        </div>
        <div class="env-stat">
          <span class="env-stat-label">Runtime Python</span>
          <span class="env-stat-value">{{
            notebook.environment.runtimePythonVersion ||
            notebook.environment.pythonVersion ||
            'Unknown'
          }}</span>
        </div>
        <div class="env-stat">
          <span class="env-stat-label">Declared</span>
          <span class="env-stat-value">{{ notebook.environment.declaredPackageCount }}</span>
        </div>
        <div class="env-stat">
          <span class="env-stat-label">Resolved</span>
          <span class="env-stat-value">{{ notebook.environment.resolvedPackageCount }}</span>
        </div>
        <div class="env-stat">
          <span class="env-stat-label">Lockfile</span>
          <span class="env-stat-value" :title="notebook.environment.lockfileHash">
            {{ shortLockfileHash }}
          </span>
        </div>
      </div>

      <div class="env-meta">
        <div>Last sync: {{ lastSyncedLabel }}</div>
        <div>Last sync duration: {{ lastSyncDurationLabel }}</div>
        <div>Runtime source: {{ interpreterSourceLabel }}</div>
        <div v-if="notebook.environment.venvPython">
          Interpreter: <code>{{ notebook.environment.venvPython }}</code>
        </div>
        <div v-else-if="!notebook.environment.hasLockfile">Lockfile not created yet</div>
      </div>

      <!--
        R-side environment summary — rendered only when the notebook
        ships a ``renv.lock``. Backed by RRuntime on the server
        (``.strata/runtime.json``). Sibling of the Python env stats
        above; intentionally a thin read-only summary in this PR.
        Package list + install-from-UI come with PR C.
      -->
      <div v-if="notebook.rEnvironment.hasLockfile" class="env-r-section">
        <div class="env-r-header">R Environment</div>
        <div class="env-stats">
          <div class="env-stat">
            <span class="env-stat-label">R version</span>
            <span class="env-stat-value">{{ notebook.rEnvironment.rVersion || 'Unknown' }}</span>
          </div>
          <div class="env-stat">
            <span class="env-stat-label">renv.lock</span>
            <span class="env-stat-value" :title="notebook.rEnvironment.lockHash">
              {{ shortRLockHash }}
            </span>
          </div>
          <div class="env-stat">
            <span class="env-stat-label">Last sync</span>
            <span class="env-stat-value">{{ rLastSyncedLabel }}</span>
          </div>
        </div>
      </div>

      <div v-if="lastActionLabel" class="env-action">
        {{ lastActionLabel }}
      </div>

      <div v-if="environmentOperation" class="env-operation">
        <div class="env-operation-header">
          <strong>{{ operationTitle }}</strong>
          <span class="env-operation-status" :class="operationStatusClass">
            {{ operationStatusLabel }}
          </span>
        </div>
        <div class="env-operation-command">
          <code>{{ environmentOperation.command }}</code>
          <span class="env-operation-duration">{{ operationDurationLabel }}</span>
        </div>
        <div v-if="operationPhaseLabel" class="env-operation-phase">
          {{ operationPhaseLabel }}
        </div>
        <details
          v-if="environmentOperation.stdout || environmentOperation.stderr"
          class="env-operation-log"
        >
          <summary>Command output</summary>
          <div v-if="environmentOperation.stdout" class="env-operation-stream">
            <strong>stdout</strong>
            <pre>{{ environmentOperation.stdout }}</pre>
            <div v-if="environmentOperation.stdoutTruncated" class="env-operation-note">
              stdout truncated for display
            </div>
          </div>
          <div v-if="environmentOperation.stderr" class="env-operation-stream">
            <strong>stderr</strong>
            <pre>{{ environmentOperation.stderr }}</pre>
            <div v-if="environmentOperation.stderrTruncated" class="env-operation-note">
              stderr truncated for display
            </div>
          </div>
        </details>
      </div>

      <div v-if="recentEnvironmentOperations.length > 0" class="env-history">
        <div class="env-history-header">
          <strong>Recent operations</strong>
        </div>
        <ul class="env-history-list">
          <li
            v-for="operation in recentEnvironmentOperations"
            :key="operation.id"
            class="env-history-item"
          >
            <div class="env-history-topline">
              <code>{{ operation.command }}</code>
              <span class="env-operation-status" :class="`operation-${operation.status}`">
                {{ operation.status }}
              </span>
            </div>
            <div class="env-history-meta">
              <span>{{
                operationTimestampLabel(operation.finishedAt || operation.startedAt)
              }}</span>
              <span v-if="operation.durationMs != null">{{ operation.durationMs }} ms</span>
              <span v-if="operation.staleCellCount > 0">
                {{ operation.staleCellCount }} cell{{ operation.staleCellCount === 1 ? '' : 's' }}
                affected
              </span>
            </div>
          </li>
        </ul>
      </div>

      <div v-if="environmentError || notebook.environment.syncError" class="env-error">
        {{ environmentError || notebook.environment.syncError }}
      </div>

      <div v-if="notebook.environment.syncNotice" class="env-notice">
        {{ notebook.environment.syncNotice }}
      </div>

      <div v-if="environmentWarnings.length > 0" class="env-warning">
        <strong>Import warnings</strong>
        <ul class="env-warning-list">
          <li v-for="warning in environmentWarnings" :key="warning">
            {{ warning }}
          </li>
        </ul>
      </div>

      <div v-if="requirementsMode" class="requirements-editor">
        <div class="requirements-header">
          <strong>{{
            requirementsMode === 'requirements-export'
              ? 'requirements.txt Export'
              : requirementsMode === 'yaml-import'
                ? 'environment.yaml Import'
                : 'requirements.txt Import'
          }}</strong>
          <button class="btn-link" @click="closeRequirementsEditor">Close</button>
        </div>
        <p class="requirements-help">
          {{
            requirementsMode === 'requirements-export'
              ? 'These are the direct notebook dependencies managed by pyproject.toml and uv.lock.'
              : requirementsMode === 'yaml-import'
                ? 'Paste or upload a Conda-style environment.yaml. Preview first; apply will replace the notebook’s direct dependencies. Channels and python pins are imported best-effort and may be ignored with warnings.'
                : 'Paste or upload plain package specifiers, one per line. Preview first; apply will replace the notebook’s direct dependencies. Comments are allowed; pip flags and nested includes are not.'
          }}
        </p>
        <div v-if="requirementsMode !== 'requirements-export'" class="requirements-file-row">
          <label class="file-picker">
            <input
              type="file"
              :accept="
                requirementsMode === 'yaml-import'
                  ? '.yaml,.yml,text/yaml,text/x-yaml'
                  : '.txt,text/plain'
              "
              @change="handleImportFile"
            />
            Load file
          </label>
          <span v-if="importFileName" class="file-name">{{ importFileName }}</span>
        </div>
        <textarea
          v-model="requirementsText"
          class="requirements-textarea"
          :readonly="requirementsMode === 'requirements-export'"
          :placeholder="
            requirementsMode === 'yaml-import'
              ? 'name: demo\ndependencies:\n  - python=3.13\n  - pyarrow=18.0.0\n  - pip:\n      - requests==2.32.3'
              : 'pandas==2.2.3\nnumpy>=2.0'
          "
        />
        <div v-if="environmentImportPreview && hasFreshImportPreview" class="import-preview">
          <div class="import-preview-summary">
            {{ environmentImportPreview.importedCount }} direct package{{
              environmentImportPreview.importedCount === 1 ? '' : 's'
            }}
            after import · {{ environmentImportPreview.additions.length }} add ·
            {{ environmentImportPreview.removals.length }} remove ·
            {{ environmentImportPreview.unchanged.length }} unchanged
          </div>

          <div v-if="environmentImportPreview.additions.length > 0" class="import-preview-block">
            <strong>Add</strong>
            <ul class="import-preview-list">
              <li
                v-for="dep in environmentImportPreview.additions"
                :key="`add-${dep.name}-${dep.specifier}`"
              >
                <code>{{ dep.name }}{{ dep.specifier || '' }}</code>
              </li>
            </ul>
          </div>

          <div v-if="environmentImportPreview.removals.length > 0" class="import-preview-block">
            <strong>Remove</strong>
            <ul class="import-preview-list">
              <li
                v-for="dep in environmentImportPreview.removals"
                :key="`remove-${dep.name}-${dep.specifier}`"
              >
                <code>{{ dep.name }}{{ dep.specifier || '' }}</code>
              </li>
            </ul>
          </div>
        </div>
        <div class="requirements-actions">
          <button
            v-if="requirementsMode === 'requirements-export'"
            class="btn-secondary"
            :disabled="!requirementsText"
            @click="downloadRequirements"
          >
            Download
          </button>
          <button
            v-if="requirementsMode !== 'requirements-export'"
            class="btn-secondary"
            :disabled="
              !connected ||
              environmentMutationActive ||
              dependencyLoading ||
              environmentLoading ||
              !requirementsText.trim()
            "
            @click="previewImport"
          >
            {{ environmentLoading ? 'Previewing…' : 'Preview Import' }}
          </button>
          <button
            v-if="requirementsMode !== 'requirements-export'"
            class="btn-sync"
            :disabled="
              !connected ||
              environmentMutationActive ||
              dependencyLoading ||
              environmentLoading ||
              !requirementsText.trim() ||
              !hasFreshImportPreview
            "
            @click="applyRequirementsImport"
          >
            {{ environmentLoading ? 'Importing…' : 'Apply Import' }}
          </button>
        </div>
      </div>

      <div class="package-view-bar">
        <div class="package-view-tabs">
          <button
            class="package-view-tab"
            :class="{ active: packageView === 'declared' }"
            @click="packageView = 'declared'"
          >
            Declared
          </button>
          <button
            class="package-view-tab"
            :class="{ active: packageView === 'resolved' }"
            @click="packageView = 'resolved'"
          >
            Resolved
          </button>
        </div>
        <input
          v-model="packageFilter"
          type="text"
          class="package-filter"
          placeholder="Filter packages"
        />
      </div>

      <div class="add-dep">
        <input
          v-model="newPackage"
          type="text"
          placeholder="Add package (e.g. pandas)"
          class="dep-input"
          :disabled="
            !connected || environmentMutationActive || dependencyLoading || environmentLoading
          "
          @keydown.enter="addPackage"
        />
        <button
          class="btn-add"
          :disabled="
            !connected ||
            environmentMutationActive ||
            dependencyLoading ||
            environmentLoading ||
            !newPackage.trim()
          "
          @click="addPackage"
        >
          {{ environmentMutationActive ? '…' : dependencyLoading ? '…' : '+' }}
        </button>
      </div>

      <div v-if="dependencyError" class="dep-error">
        {{ dependencyError }}
      </div>

      <div v-if="currentPackageList.length === 0" class="dep-empty">
        {{ packageView === 'declared' ? 'No declared packages' : 'No resolved packages' }}
      </div>
      <ul v-else class="dep-list">
        <li
          v-for="dep in currentPackageList"
          :key="`${packageView}-${dep.name}-${dep.specifier}-${dep.version}`"
          class="dep-item"
        >
          <div class="dep-details">
            <span class="dep-name">{{ dep.name }}</span>
            <span class="dep-version">{{
              packageView === 'resolved'
                ? dep.version || 'unknown'
                : dep.version || dep.specifier || 'unpinned'
            }}</span>
          </div>
          <button
            v-if="packageView === 'declared'"
            class="btn-remove"
            title="Remove"
            :disabled="environmentMutationActive || dependencyLoading || environmentLoading"
            @click="removePackage(dep.name)"
          >
            Remove
          </button>
        </li>
      </ul>
    </div>

    <PythonVersionModal
      :open="pythonModalOpen"
      :available="pythonAvailable"
      :current="notebook.environment.requestedPythonVersion || ''"
      :busy="pythonModalBusy"
      :error-message="pythonModalError"
      @close="closePythonModal"
      @confirm="confirmPythonChange"
    />
  </div>
</template>

<style scoped>
.env-panel {
  margin-top: 12px;
  border-top: 1px solid var(--border-subtle);
  padding-top: 8px;
}

.env-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  background: none;
  border: none;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  cursor: pointer;
  padding: 4px 0;
}

.env-toggle:hover {
  color: var(--text-primary);
}

.dep-count {
  background: var(--bg-input);
  color: var(--accent-primary);
  padding: 1px 6px;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
}

.toggle-icon {
  margin-left: auto;
  font-size: 10px;
}

.env-content {
  margin-top: 8px;
}

.env-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}

.env-status-block {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.env-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--text-primary);
  font-size: 12px;
  font-weight: 600;
}

.env-status-help {
  color: var(--text-secondary);
  font-size: 11px;
  line-height: 1.4;
  max-width: 320px;
}

.env-status-help code {
  color: var(--accent-primary);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--text-muted);
}

.state-ready {
  background: var(--accent-success);
}

.state-pending {
  background: var(--accent-primary);
}

.state-fallback {
  background: var(--accent-warning);
}

.state-failed {
  background: var(--accent-danger);
}

.state-unknown {
  background: var(--text-muted);
}

.env-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}

.btn-sync {
  border: 1px solid var(--border-strong);
  background: var(--tint-primary);
  color: var(--text-primary);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 11px;
  cursor: pointer;
}

.btn-secondary {
  border: 1px solid var(--border-strong);
  background: transparent;
  color: var(--text-primary);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 11px;
  cursor: pointer;
}

.btn-sync:hover,
.btn-secondary:hover {
  border-color: var(--accent-primary);
  color: var(--accent-primary);
}

.btn-sync:disabled,
.btn-secondary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.env-stats {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
  margin-bottom: 8px;
}

.env-stat {
  background: var(--tint-primary);
  border: 1px solid var(--bg-input);
  border-radius: 8px;
  padding: 8px;
}

.env-stat-label {
  display: block;
  color: var(--text-muted);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  margin-bottom: 3px;
}

.env-stat-value {
  color: var(--text-primary);
  font-size: 12px;
  font-weight: 600;
  word-break: break-all;
}
.env-stat-button {
  background: transparent;
  border: 1px solid transparent;
  padding: 1px 6px;
  margin-left: -6px;
  border-radius: 4px;
  cursor: pointer;
  text-align: left;
  font: inherit;
  color: inherit;
}
.env-stat-button:hover:not(:disabled) {
  border-color: var(--border-strong);
  background: var(--bg-input);
}
.env-stat-button:disabled {
  cursor: not-allowed;
  opacity: 0.7;
}

.env-meta {
  color: var(--text-secondary);
  font-size: 11px;
  margin-bottom: 8px;
}

.env-meta code {
  color: var(--accent-primary);
  word-break: break-all;
}

.env-r-section {
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px dashed var(--border-subtle);
}

.env-r-header {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 6px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.env-action {
  color: var(--accent-success);
  font-size: 11px;
  margin-bottom: 8px;
  padding: 6px 8px;
  background: rgb(166 227 161 / 10%);
  border: 1px solid rgb(166 227 161 / 18%);
  border-radius: 6px;
}

.env-operation {
  color: var(--text-primary);
  font-size: 11px;
  margin-bottom: 8px;
  padding: 8px;
  background: var(--tint-primary);
  border: 1px solid var(--bg-input);
  border-radius: 8px;
}

.env-operation-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
}

.env-operation-status {
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.operation-running {
  background: rgb(137 180 250 / 14%);
  color: var(--accent-primary);
}

.operation-completed {
  background: rgb(166 227 161 / 14%);
  color: var(--accent-success);
}

.operation-failed {
  background: rgb(243 139 168 / 14%);
  color: var(--accent-danger);
}

.env-operation-command {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.env-operation-command code {
  color: var(--accent-primary);
  word-break: break-all;
}

.env-operation-duration {
  color: var(--text-secondary);
}

.env-operation-log {
  margin-top: 8px;
  border-top: 1px solid var(--bg-input);
  padding-top: 8px;
}

.env-operation-log summary {
  cursor: pointer;
  color: var(--text-secondary);
}

.env-operation-stream {
  margin-top: 8px;
}

.env-operation-stream strong {
  display: block;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.env-operation-stream pre {
  margin: 0;
  padding: 8px;
  max-height: 160px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  border-radius: 6px;
  border: 1px solid var(--bg-input);
  background: var(--tint-primary);
  color: var(--text-primary);
}

.env-operation-note {
  margin-top: 4px;
  color: var(--accent-warning);
}

.env-history {
  color: var(--text-primary);
  font-size: 11px;
  margin-bottom: 8px;
  padding: 8px;
  background: var(--tint-primary);
  border: 1px solid var(--bg-input);
  border-radius: 8px;
}

.env-history-header {
  margin-bottom: 6px;
}

.env-history-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.env-history-item {
  border-top: 1px solid var(--bg-input);
  padding-top: 6px;
}

.env-history-item:first-child {
  border-top: 0;
  padding-top: 0;
}

.env-history-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.env-history-topline code {
  color: var(--accent-primary);
  word-break: break-all;
}

.env-history-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 4px;
  color: var(--text-secondary);
}

.env-error,
.dep-error {
  color: var(--accent-danger);
  font-size: 11px;
  margin-bottom: 6px;
  padding: 6px 8px;
  background: rgb(243 139 168 / 10%);
  border: 1px solid rgb(243 139 168 / 18%);
  border-radius: 6px;
}

.env-warning,
.env-notice {
  color: var(--accent-warning);
  font-size: 11px;
  margin-bottom: 6px;
  padding: 6px 8px;
  background: rgb(249 226 175 / 10%);
  border: 1px solid rgb(249 226 175 / 18%);
  border-radius: 6px;
}

.env-warning-list {
  margin-top: 4px;
  padding-left: 16px;
}

.requirements-editor {
  margin-bottom: 10px;
  padding: 8px;
  border: 1px solid var(--bg-input);
  border-radius: 8px;
  background: var(--tint-primary);
}

.requirements-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  color: var(--text-primary);
  font-size: 12px;
  margin-bottom: 6px;
}

.requirements-help {
  color: var(--text-secondary);
  font-size: 11px;
  margin-bottom: 8px;
}

.requirements-file-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.file-picker {
  position: relative;
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--border-strong);
  background: transparent;
  color: var(--text-primary);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 11px;
  cursor: pointer;
}

.file-picker input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}

.file-name {
  color: var(--text-secondary);
  font-size: 11px;
  word-break: break-all;
}

.requirements-textarea {
  width: 100%;
  min-height: 120px;
  resize: vertical;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid var(--border-strong);
  background: var(--tint-primary);
  color: var(--text-primary);
  font: inherit;
  margin-bottom: 8px;
}

.requirements-textarea:focus {
  outline: none;
  border-color: var(--accent-primary);
}

.import-preview {
  margin-bottom: 8px;
  padding: 8px;
  border: 1px solid var(--bg-input);
  background: var(--tint-primary);
  border-radius: 8px;
}

.import-preview-summary {
  color: var(--text-primary);
  font-size: 11px;
  font-weight: 600;
  margin-bottom: 6px;
}

.import-preview-block {
  margin-top: 6px;
  color: var(--text-secondary);
  font-size: 11px;
}

.import-preview-list {
  margin: 4px 0 0;
  padding-left: 16px;
}

.requirements-actions {
  display: flex;
  justify-content: flex-end;
  gap: 6px;
}

.btn-link {
  background: none;
  border: none;
  color: var(--accent-primary);
  cursor: pointer;
  padding: 0;
  font-size: 11px;
}

.btn-link:hover {
  text-decoration: underline;
}

.package-view-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin: 10px 0 8px;
}

.package-view-tabs {
  display: inline-flex;
  gap: 6px;
}

.package-view-tab {
  border: 1px solid var(--bg-input);
  background: transparent;
  color: var(--text-secondary);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 11px;
  cursor: pointer;
}

.package-view-tab.active {
  border-color: var(--accent-primary);
  color: var(--accent-primary);
  background: rgb(137 180 250 / 10%);
}

.package-filter {
  flex: 1;
  min-width: 0;
  border: 1px solid var(--bg-input);
  background: var(--tint-primary);
  color: var(--text-primary);
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12px;
}

.add-dep {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
}

.dep-input {
  flex: 1;
  min-width: 0;
  background: var(--tint-primary);
  border: 1px solid var(--bg-input);
  border-radius: 6px;
  padding: 8px 10px;
  color: var(--text-primary);
  font-size: 12px;
}

.dep-input:focus {
  outline: none;
  border-color: var(--accent-primary);
}

.btn-add {
  border: 1px solid var(--border-strong);
  background: var(--tint-primary);
  color: var(--text-primary);
  border-radius: 6px;
  min-width: 36px;
  padding: 0 10px;
  cursor: pointer;
}

.btn-add:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.dep-empty {
  color: var(--text-muted);
  font-size: 11px;
  padding: 6px 0;
}

.dep-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.dep-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px;
  border: 1px solid var(--bg-input);
  border-radius: 8px;
  background: var(--tint-primary);
}

.dep-details {
  min-width: 0;
}

.dep-name {
  display: block;
  color: var(--text-primary);
  font-size: 12px;
  font-weight: 600;
}

.dep-version {
  display: block;
  color: var(--text-secondary);
  font-size: 11px;
  word-break: break-all;
}

.btn-remove {
  border: 1px solid var(--border-strong);
  background: transparent;
  color: var(--accent-danger);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 11px;
  cursor: pointer;
}

.btn-remove:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
