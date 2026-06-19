<script setup lang="ts">
import { computed } from 'vue'
import { useNotebook } from '../stores/notebook'
import type { CellId } from '../types/notebook'

const props = defineProps<{ cellId: CellId }>()

const { cellMap, runCellTests, updateTestSource, closeTests } = useNotebook()

const cell = computed(() => cellMap.value.get(props.cellId))
const result = computed(() => cell.value?.testResult ?? null)
const running = computed(() => cell.value?.testStatus === 'running')

const cellLabel = computed(() => {
  const c = cell.value
  if (c?.annotations?.name) return c.annotations.name
  return props.cellId.slice(0, 8)
})

const testSource = computed({
  get: () => cell.value?.testSource ?? '',
  set: (v: string) => updateTestSource(props.cellId, v),
})

const summary = computed(() => {
  const r = result.value
  if (!r) return ''
  const parts: string[] = []
  if (r.passed) parts.push(`${r.passed} passed`)
  if (r.failed) parts.push(`${r.failed} failed`)
  if (r.errored) parts.push(`${r.errored} errored`)
  if (r.skipped) parts.push(`${r.skipped} skipped`)
  return parts.length ? parts.join(' · ') : 'no tests'
})

const summaryClass = computed(() => {
  const r = result.value
  if (!r) return ''
  if (r.errored > 0) return 'summary--error'
  if (r.failed > 0) return 'summary--fail'
  return 'summary--pass'
})

const ranAtLabel = computed(() => {
  const r = result.value
  if (!r || !r.ranAt) return ''
  return new Date(r.ranAt).toLocaleTimeString()
})

function outcomeMark(outcome: string): string {
  if (outcome === 'passed') return '✓'
  if (outcome === 'failed') return '✗'
  if (outcome === 'skipped') return '○'
  return '!'
}

function run() {
  if (!running.value) runCellTests(props.cellId)
}

function handleKeydown(e: KeyboardEvent) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault()
    run()
  }
}
</script>

<template>
  <div class="tests-panel">
    <div class="tests-header">
      <span class="tests-title">
        Tests for <code>{{ cellLabel }}</code>
      </span>
      <span v-if="running" class="tests-running">running…</span>
      <button class="tests-close-btn" title="Close tests panel" @click="closeTests(cellId)">
        &times;
      </button>
    </div>

    <div class="tests-editor-wrap">
      <textarea
        v-model="testSource"
        class="tests-editor"
        spellcheck="false"
        placeholder="def test_x(cell):  # cell.X = a def or upstream input&#10;    assert cell.featurize(cell.trips).num_rows == 100"
        @keydown="handleKeydown"
      ></textarea>
      <div class="tests-editor-actions">
        <span class="tests-hint">Cmd/Ctrl+Enter to run</span>
        <button class="tests-run-btn" :disabled="running" @click="run">
          {{ running ? 'Running…' : 'Run tests' }}
        </button>
      </div>
    </div>

    <div class="tests-results">
      <div v-if="result?.pytestUnavailable" class="tests-callout">
        Cell tests need <code>pytest</code> in this notebook's environment — add it from the
        dependencies panel, then run again.
      </div>

      <template v-else-if="result">
        <div class="tests-summary-row">
          <span class="tests-summary" :class="summaryClass">{{ summary }}</span>
          <span
            v-if="result.stale"
            class="tests-stale-chip"
            title="Cell or tests changed since run"
          >
            stale
          </span>
          <span v-if="ranAtLabel" class="tests-ran-at">{{ ranAtLabel }}</span>
        </div>

        <ul class="tests-list">
          <li
            v-for="t in result.tests"
            :key="t.nodeid || t.name"
            class="tests-row"
            :class="`tests-row--${t.outcome}`"
          >
            <div class="tests-row-head">
              <span class="tests-mark">{{ outcomeMark(t.outcome) }}</span>
              <span class="tests-name">{{ t.name }}</span>
            </div>
            <pre
              v-if="t.message && (t.outcome === 'failed' || t.outcome === 'error')"
              class="tests-message"
              >{{ t.message }}</pre
            >
          </li>
        </ul>
      </template>

      <div v-else class="tests-empty">No tests run yet.</div>
    </div>
  </div>
</template>

<style scoped>
.tests-panel {
  background: var(--bg-base);
  border: 1px solid var(--accent-primary);
  border-radius: 8px;
  margin-top: 4px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  max-height: 420px;
}
.tests-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-bottom: 1px solid var(--border-subtle);
  font-size: 12px;
  background: var(--bg-surface);
}
.tests-title {
  color: var(--accent-primary);
  font-weight: 600;
}
.tests-title code {
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.tests-running {
  color: var(--accent-warning);
  font-size: 10px;
  animation: pulse 1s infinite;
}
@keyframes pulse {
  50% {
    opacity: 0.4;
  }
}
.tests-close-btn {
  margin-left: auto;
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 16px;
  padding: 0 4px;
  line-height: 1;
}
.tests-close-btn:hover {
  color: var(--accent-danger);
}

.tests-editor-wrap {
  border-bottom: 1px solid var(--border-subtle);
  background: var(--bg-surface);
}
.tests-editor {
  width: 100%;
  min-height: 88px;
  resize: vertical;
  box-sizing: border-box;
  background: var(--bg-elevated);
  border: none;
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  line-height: 1.5;
  padding: 8px 12px;
  outline: none;
}
.tests-editor::placeholder {
  color: var(--border-strong);
}
.tests-editor-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
}
.tests-hint {
  color: var(--text-muted);
  font-size: 10px;
}
.tests-run-btn {
  margin-left: auto;
  background: var(--tint-primary);
  color: var(--accent-primary);
  border: 1px solid var(--accent-primary);
  border-radius: 4px;
  padding: 4px 12px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}
.tests-run-btn:hover:not(:disabled) {
  background: var(--tint-primary-strong);
}
.tests-run-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.tests-results {
  flex: 1;
  overflow-y: auto;
  padding: 8px 12px;
  font-size: 12px;
  min-height: 40px;
}
.tests-empty {
  color: var(--text-muted);
  font-size: 11px;
  font-style: italic;
}
.tests-callout {
  color: var(--accent-warning);
  font-size: 11px;
  line-height: 1.5;
}
.tests-callout code {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  color: var(--text-primary);
}
.tests-summary-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.tests-summary {
  font-weight: 600;
}
.summary--pass {
  color: var(--accent-success);
}
.summary--fail {
  color: var(--accent-danger);
}
.summary--error {
  color: var(--accent-warning);
}
.tests-stale-chip {
  background: var(--bg-elevated);
  color: var(--text-muted);
  border: 1px solid var(--border-subtle);
  border-radius: 3px;
  padding: 0 6px;
  font-size: 10px;
}
.tests-ran-at {
  margin-left: auto;
  color: var(--text-muted);
  font-size: 10px;
}
.tests-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.tests-row {
  padding: 3px 0;
  border-top: 1px solid var(--border-subtle);
}
.tests-row-head {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.tests-mark {
  width: 14px;
  text-align: center;
  font-weight: 700;
}
.tests-row--passed .tests-mark {
  color: var(--accent-success);
}
.tests-row--failed .tests-mark {
  color: var(--accent-danger);
}
.tests-row--error .tests-mark {
  color: var(--accent-warning);
}
.tests-row--skipped .tests-mark {
  color: var(--text-muted);
}
.tests-name {
  color: var(--text-primary);
}
.tests-message {
  margin: 4px 0 4px 20px;
  padding: 6px 8px;
  background: var(--bg-elevated);
  border-radius: 4px;
  color: var(--accent-danger);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 160px;
  overflow-y: auto;
}
.tests-row--error .tests-message {
  color: var(--accent-warning);
}
</style>
