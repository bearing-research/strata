<script setup lang="ts">
import { GitBranch, RefreshCw, X } from '@lucide/vue'
import { computed, onMounted, ref, watch } from 'vue'
import { useNotebook } from '../stores/notebook'
import type { CellId } from '../types/notebook'
import {
  branchOrigin,
  checkpointContents,
  type CheckpointInfo,
  type CheckpointRunInfo,
  type CheckpointRunStatus,
} from '../utils/checkpoints'
import { formatBytes } from '../utils/inputWeights'

const props = defineProps<{ cellId: CellId }>()
const emit = defineEmits<{ close: [] }>()

const { cellMap, connected, environmentMutationActive, executeCellWebSocket, listCellCheckpoints } =
  useNotebook()

const runs = ref<CheckpointRunInfo[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

const cell = computed(() => cellMap.value.get(props.cellId))
const running = computed(() => cell.value?.status === 'running')
const canBranch = computed(
  () => connected.value && !environmentMutationActive.value && !running.value,
)

const cellLabel = computed(() => cell.value?.annotations?.name || props.cellId.slice(0, 8))

// A run's start and its end each refresh, and the earlier answer can land last.
let latestRefresh = 0

async function refresh() {
  const mine = ++latestRefresh
  loading.value = true
  try {
    const listed = await listCellCheckpoints(props.cellId)
    if (mine !== latestRefresh) return
    runs.value = listed
    error.value = null
  } catch (err) {
    if (mine !== latestRefresh) return
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    if (mine === latestRefresh) loading.value = false
  }
}

onMounted(refresh)
// A run that starts or settles changes what is stored; checkpoints saved meanwhile
// arrive with Refresh.
watch(() => cell.value?.status, refresh)

function branch(checkpoint: CheckpointInfo) {
  if (canBranch.value) executeCellWebSocket(props.cellId, checkpoint.releaseId)
}

function formatTime(seconds: number): string {
  return new Date(seconds * 1000).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

const STATUS_TITLES: Record<CheckpointRunStatus, string> = {
  running: 'This run is still going',
  interrupted:
    'This run stopped before it finished. Run continues it from its latest checkpoint while the cell and its inputs are unchanged.',
  finished: 'This run finished and stored its outputs',
  failed: 'This run was ended without finishing',
}

function originLabel(run: CheckpointRunInfo): string {
  if (run.init === null) return ''
  const origin = branchOrigin(run, runs.value)
  if (!origin) return 'branch'
  const from =
    origin.run.startedAt === null ? '' : ` of the ${formatTime(origin.run.startedAt)} run`
  return `branch from #${origin.checkpoint.sequence}${from}`
}
</script>

<template>
  <div class="checkpoints-panel" data-testid="checkpoints-panel">
    <div class="checkpoints-header">
      <span class="checkpoints-title">
        Checkpoints of <code>{{ cellLabel }}</code>
      </span>
      <span v-if="loading" class="checkpoints-loading">loading…</span>
      <button
        class="checkpoints-icon-btn checkpoints-refresh"
        title="Refresh"
        aria-label="Refresh checkpoints"
        :disabled="loading"
        @click="refresh"
      >
        <RefreshCw />
      </button>
      <button
        class="checkpoints-icon-btn checkpoints-close"
        title="Close checkpoints panel"
        aria-label="Close checkpoints panel"
        @click="emit('close')"
      >
        <X />
      </button>
    </div>

    <div class="checkpoints-body">
      <div v-if="error" class="checkpoints-error">{{ error }}</div>
      <div v-else-if="!runs.length" class="checkpoints-empty">
        <template v-if="!loading">
          No checkpoints stored. A cell saves one with <code>strata.checkpoint(...)</code>.
        </template>
      </div>
      <template v-else>
        <p class="checkpoints-hint">
          A branch runs this cell again from a checkpoint, read back as on a resume
          (<code>strata.restore()</code>). It is its own run: it never serves or overwrites the
          ordinary run's cache.
        </p>
        <section
          v-for="run in runs"
          :key="run.buildId"
          class="checkpoints-run"
          data-testid="checkpoints-run"
        >
          <div class="checkpoints-run-head">
            <span class="checkpoints-run-title">
              Run<template v-if="run.startedAt !== null">
                of {{ formatTime(run.startedAt) }}</template
              >
            </span>
            <span v-if="run.variant" class="checkpoints-chip" title="Variant">{{
              run.variant
            }}</span>
            <span
              v-if="run.status"
              class="checkpoints-status"
              :class="`checkpoints-status--${run.status}`"
              :title="STATUS_TITLES[run.status]"
            >
              {{ run.status }}
            </span>
            <span v-if="run.init !== null" class="checkpoints-origin">{{ originLabel(run) }}</span>
          </div>
          <ul class="checkpoints-list">
            <li
              v-for="checkpoint in run.checkpoints"
              :key="checkpoint.releaseId"
              class="checkpoints-row"
              data-testid="checkpoint-row"
            >
              <span class="checkpoints-seq">#{{ checkpoint.sequence }}</span>
              <span class="checkpoints-time">{{ formatTime(checkpoint.createdAt) }}</span>
              <span class="checkpoints-contents" :title="checkpointContents(checkpoint)">
                {{ checkpointContents(checkpoint) }}
              </span>
              <span v-if="checkpoint.step !== null" class="checkpoints-chip">
                step {{ checkpoint.step }}
              </span>
              <span
                v-for="head in checkpoint.heads"
                :key="head"
                class="checkpoints-chip checkpoints-chip--head"
                title="A head points at this checkpoint, so prune keeps it"
              >
                {{ head }}
              </span>
              <span class="checkpoints-size">{{ formatBytes(checkpoint.byteSize) }}</span>
              <button
                class="checkpoints-branch-btn"
                title="Run this cell from this checkpoint"
                data-testid="checkpoint-branch-button"
                :disabled="!canBranch"
                @click="branch(checkpoint)"
              >
                <GitBranch /> Branch from here
              </button>
            </li>
          </ul>
        </section>
      </template>
    </div>
  </div>
</template>

<style scoped>
.checkpoints-panel {
  background: var(--bg-base);
  border: 1px solid var(--accent-primary);
  border-radius: 8px;
  margin-top: 4px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  max-height: 420px;
}
.checkpoints-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-bottom: 1px solid var(--border-subtle);
  font-size: 12px;
  background: var(--bg-surface);
}
.checkpoints-title {
  color: var(--accent-primary);
  font-weight: 600;
}
.checkpoints-panel code {
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.checkpoints-loading {
  color: var(--text-muted);
  font-size: 10px;
}
.checkpoints-icon-btn {
  display: inline-flex;
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 0 4px;
  line-height: 1;
}
.checkpoints-icon-btn :deep(svg) {
  width: 14px;
  height: 14px;
}
.checkpoints-refresh {
  margin-left: auto;
}
.checkpoints-refresh:hover:not(:disabled) {
  color: var(--accent-primary);
}
.checkpoints-refresh:disabled {
  opacity: 0.4;
  cursor: default;
}
.checkpoints-close:hover {
  color: var(--accent-danger);
}
.checkpoints-body {
  flex: 1;
  overflow-y: auto;
  padding: 8px 12px;
  font-size: 12px;
  min-height: 40px;
}
.checkpoints-empty,
.checkpoints-hint {
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.5;
  margin: 0;
}
.checkpoints-error {
  color: var(--accent-danger);
  font-size: 11px;
}
.checkpoints-run {
  margin-top: 10px;
}
.checkpoints-run-head {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding-bottom: 4px;
}
.checkpoints-run-title {
  color: var(--text-primary);
  font-weight: 600;
}
.checkpoints-status {
  font-size: 11px;
  cursor: help;
}
.checkpoints-status--running {
  color: var(--accent-primary);
}
.checkpoints-status--interrupted {
  color: var(--accent-warning);
}
.checkpoints-status--finished {
  color: var(--accent-success);
}
.checkpoints-status--failed {
  color: var(--accent-danger);
}
.checkpoints-origin {
  color: var(--text-muted);
  font-size: 11px;
}
.checkpoints-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.checkpoints-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 0;
  border-top: 1px solid var(--border-subtle);
}
.checkpoints-seq {
  min-width: 28px;
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-weight: 600;
}
.checkpoints-time,
.checkpoints-size {
  color: var(--text-muted);
  font-size: 11px;
  white-space: nowrap;
}
.checkpoints-contents {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 11px;
}
.checkpoints-chip {
  background: var(--bg-elevated);
  color: var(--text-muted);
  border: 1px solid var(--border-subtle);
  border-radius: 3px;
  padding: 0 6px;
  font-size: 10px;
  white-space: nowrap;
}
.checkpoints-chip--head {
  color: var(--accent-success);
  border-color: var(--accent-success);
  cursor: help;
}
.checkpoints-branch-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--tint-primary);
  color: var(--accent-primary);
  border: 1px solid var(--accent-primary);
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
  cursor: pointer;
}
.checkpoints-branch-btn :deep(svg) {
  width: 12px;
  height: 12px;
}
.checkpoints-branch-btn:hover:not(:disabled) {
  background: var(--tint-primary-strong);
}
.checkpoints-branch-btn:disabled {
  opacity: 0.4;
  cursor: default;
}
</style>
