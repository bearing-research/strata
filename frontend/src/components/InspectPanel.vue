<script setup lang="ts">
import { ref, nextTick, watch, computed, onMounted } from 'vue'
import { useNotebook } from '../stores/notebook'
import type { CellId } from '../types/notebook'
import { useStrata, type CellIterationInfo } from '../composables/useStrata'

const props = defineProps<{ cellId: CellId }>()

const { inspectReadyFor, inspectHistoryFor, evalInspect, closeInspect, cellMap, notebook } =
  useNotebook()
const strata = useStrata()

const inputExpr = ref('')
const historyEl = ref<HTMLElement | null>(null)

const ready = computed(() => inspectReadyFor(props.cellId))
const history = computed(() => inspectHistoryFor(props.cellId))

const iterations = ref<CellIterationInfo[]>([])
const selectedIteration = ref<number | null>(null)
const iterationsError = ref<string | null>(null)
const iterationsLoading = ref(false)
const uriCopiedHint = ref(false)

const currentCell = computed(() => cellMap.value.get(props.cellId))
const carryVariable = computed(() => currentCell.value?.annotations?.loop?.carry ?? null)

const cellLabel = computed(() => {
  const cell = currentCell.value
  // @name, else the short id. Not a define name: the REPL sees the cell's inputs.
  if (cell?.annotations?.name) return cell.annotations.name
  return props.cellId.slice(0, 8)
})

const selectedIterationInfo = computed(
  () => iterations.value.find((entry) => entry.iteration === selectedIteration.value) ?? null,
)

async function refreshIterations() {
  if (!carryVariable.value) {
    iterations.value = []
    selectedIteration.value = null
    return
  }
  const notebookId = notebook.id
  if (!notebookId) return
  iterationsLoading.value = true
  iterationsError.value = null
  try {
    const result = await strata.listCellIterations(notebookId, props.cellId)
    iterations.value = result.iterations
    if (!iterations.value.length) {
      selectedIteration.value = null
      return
    }
    const last = iterations.value[iterations.value.length - 1].iteration
    if (
      selectedIteration.value === null ||
      !iterations.value.some((entry) => entry.iteration === selectedIteration.value)
    ) {
      selectedIteration.value = last
    }
  } catch (err) {
    iterationsError.value = err instanceof Error ? err.message : String(err)
  } finally {
    iterationsLoading.value = false
  }
}

onMounted(() => {
  refreshIterations()
})

// Refresh as iterations complete so the picker tracks live progress.
watch(
  () => currentCell.value?.loopProgress?.iteration ?? -1,
  () => {
    refreshIterations()
  },
)

watch(
  () => props.cellId,
  () => {
    selectedIteration.value = null
    refreshIterations()
  },
)

async function copyArtifactUri() {
  const info = selectedIterationInfo.value
  if (!info) return
  try {
    await navigator.clipboard.writeText(info.artifactUri)
    uriCopiedHint.value = true
    setTimeout(() => {
      uriCopiedHint.value = false
    }, 1500)
  } catch {
    // Clipboard API can fail in insecure contexts.
  }
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

watch(
  () => history.value.length,
  async () => {
    await nextTick()
    if (historyEl.value) {
      historyEl.value.scrollTop = historyEl.value.scrollHeight
    }
  },
)

function submitExpr() {
  const expr = inputExpr.value.trim()
  if (!expr) return
  evalInspect(props.cellId, expr)
  inputExpr.value = ''
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    submitExpr()
  }
}

function close() {
  closeInspect(props.cellId)
}
</script>

<template>
  <div class="inspect-panel">
    <div class="inspect-header">
      <span class="inspect-title">
        Inspect inputs of <code>{{ cellLabel }}</code>
      </span>
      <span v-if="!ready" class="inspect-loading">loading...</span>
      <button class="inspect-close-btn" title="Close inspect REPL" @click="close">&times;</button>
    </div>

    <div v-if="carryVariable" class="iteration-picker">
      <div class="iteration-picker-header">
        <span class="iteration-picker-title">
          Iterations of <code>{{ carryVariable }}</code>
        </span>
        <span v-if="iterationsLoading" class="iteration-picker-loading">refreshing...</span>
      </div>
      <div v-if="iterationsError" class="iteration-picker-error">
        {{ iterationsError }}
      </div>
      <div v-else-if="!iterations.length" class="iteration-picker-empty">
        No iteration artifacts yet. Run the cell to populate iterations.
      </div>
      <div v-else class="iteration-picker-row">
        <select
          v-model.number="selectedIteration"
          class="iteration-picker-select"
          title="Select an iteration to inspect"
        >
          <option v-for="entry in iterations" :key="entry.iteration" :value="entry.iteration">
            iter {{ entry.iteration }} · {{ formatBytes(entry.byteSize) }}
          </option>
        </select>
        <button
          v-if="selectedIterationInfo"
          class="iteration-picker-copy"
          :title="`Copy ${selectedIterationInfo.artifactUri}`"
          @click="copyArtifactUri"
        >
          {{ uriCopiedHint ? 'copied' : 'copy URI' }}
        </button>
      </div>
      <div v-if="selectedIterationInfo" class="iteration-picker-meta">
        <span class="iteration-picker-pill">{{ selectedIterationInfo.contentType }}</span>
        <span v-if="selectedIterationInfo.rowCount !== null" class="iteration-picker-pill">
          {{ selectedIterationInfo.rowCount }} rows
        </span>
        <code class="iteration-picker-uri">{{ selectedIterationInfo.artifactUri }}</code>
      </div>
    </div>

    <div ref="historyEl" class="inspect-history">
      <div v-if="!history.length && ready" class="inspect-hint">
        Type an expression to inspect cell inputs. Variables from upstream cells are pre-loaded.
      </div>
      <div v-for="(entry, i) in history" :key="i" class="inspect-entry">
        <div class="inspect-expr">
          <span class="prompt-marker">&gt;&gt;&gt;</span> {{ entry.expr }}
        </div>
        <div v-if="entry.stdout" class="inspect-stdout">{{ entry.stdout }}</div>
        <div v-if="entry.error" class="inspect-error">{{ entry.error }}</div>
        <div v-else-if="entry.result" class="inspect-result">
          <span v-if="entry.type" class="inspect-type">{{ entry.type }}</span>
          <pre class="inspect-value">{{ entry.result }}</pre>
        </div>
      </div>
    </div>

    <div class="inspect-input-row">
      <span class="prompt-marker">&gt;&gt;&gt;</span>
      <input
        v-model="inputExpr"
        class="inspect-input"
        placeholder="expression..."
        :disabled="!ready"
        @keydown="handleKeydown"
      />
      <button class="inspect-run-btn" :disabled="!ready || !inputExpr.trim()" @click="submitExpr">
        Eval
      </button>
    </div>
  </div>
</template>

<style scoped>
.inspect-panel {
  background: var(--bg-base);
  border: 1px solid var(--accent-primary);
  border-radius: 8px;
  margin-top: 4px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  max-height: 360px;
}
.inspect-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-bottom: 1px solid var(--border-subtle);
  font-size: 12px;
  background: var(--bg-surface);
}
.inspect-title {
  color: var(--accent-primary);
  font-weight: 600;
}
.inspect-title code {
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.inspect-loading {
  color: var(--accent-warning);
  font-size: 10px;
  animation: pulse 1s infinite;
}
@keyframes pulse {
  50% {
    opacity: 0.4;
  }
}
.inspect-close-btn {
  margin-left: auto;
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 16px;
  padding: 0 4px;
  line-height: 1;
}
.inspect-close-btn:hover {
  color: var(--accent-danger);
}

.inspect-history {
  flex: 1;
  overflow-y: auto;
  padding: 8px 12px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  min-height: 60px;
  max-height: 260px;
}
.inspect-hint {
  color: var(--text-muted);
  font-size: 11px;
  font-family: inherit;
}
.inspect-entry {
  margin-bottom: 8px;
}
.inspect-expr {
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-all;
}
.prompt-marker {
  color: var(--accent-primary);
  user-select: none;
  margin-right: 4px;
}
.inspect-stdout {
  color: var(--text-secondary);
  white-space: pre-wrap;
  padding-left: 24px;
}
.inspect-error {
  color: var(--accent-danger);
  white-space: pre-wrap;
  padding-left: 24px;
  font-size: 11px;
}
.inspect-result {
  padding-left: 24px;
}
.inspect-type {
  color: var(--text-muted);
  font-size: 10px;
  display: block;
  margin-bottom: 1px;
}
.inspect-value {
  color: var(--accent-success);
  white-space: pre-wrap;
  margin: 0;
  font-size: 12px;
  max-height: 120px;
  overflow-y: auto;
}

.inspect-input-row {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  border-top: 1px solid var(--border-subtle);
  background: var(--bg-surface);
}
.inspect-input {
  flex: 1;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  padding: 4px 8px;
  outline: none;
}
.inspect-input:focus {
  border-color: var(--accent-primary);
}
.inspect-input::placeholder {
  color: var(--border-strong);
}
.inspect-input:disabled {
  opacity: 0.5;
}
.inspect-run-btn {
  background: var(--tint-primary);
  color: var(--accent-primary);
  border: 1px solid var(--accent-primary);
  border-radius: 4px;
  padding: 4px 10px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}
.inspect-run-btn:hover:not(:disabled) {
  background: var(--tint-primary-strong);
}
.inspect-run-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.iteration-picker {
  padding: 6px 12px;
  border-bottom: 1px solid var(--border-subtle);
  background: var(--bg-surface);
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.iteration-picker-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
}
.iteration-picker-title {
  color: var(--accent-primary);
  font-weight: 600;
}
.iteration-picker-title code {
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.iteration-picker-loading {
  color: var(--accent-warning);
  font-size: 10px;
}
.iteration-picker-error {
  color: var(--accent-danger);
  font-size: 11px;
}
.iteration-picker-empty {
  color: var(--text-muted);
  font-size: 11px;
  font-style: italic;
}
.iteration-picker-row {
  display: flex;
  align-items: center;
  gap: 6px;
}
.iteration-picker-select {
  flex: 1;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  padding: 3px 6px;
  outline: none;
}
.iteration-picker-select:focus {
  border-color: var(--accent-primary);
}
.iteration-picker-copy {
  background: var(--tint-primary);
  color: var(--accent-primary);
  border: 1px solid var(--accent-primary);
  border-radius: 4px;
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}
.iteration-picker-copy:hover {
  background: var(--tint-primary-strong);
}
.iteration-picker-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  font-size: 10px;
}
.iteration-picker-pill {
  background: var(--bg-base);
  color: var(--text-secondary);
  padding: 1px 6px;
  border-radius: 3px;
  border: 1px solid var(--border-subtle);
}
.iteration-picker-uri {
  color: var(--text-muted);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 10px;
  word-break: break-all;
}
</style>
