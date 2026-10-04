<script setup lang="ts">
import { ref, watch } from 'vue'
import type { EditableWorkerSpec, WorkerSpec } from '../types/notebook'
import { editorWorkerTransport } from '../utils/notebookWorkers'

interface WorkerDraft {
  localId: string
  persistedName: string
  name: string
  backend: WorkerSpec['backend']
  editing: boolean
  enabled: boolean
  runtimeId: string
  executorUrl: string
  transport: 'direct' | 'signed'
  strataUrl: string
  extraConfig: Record<string, unknown>
}

const props = withDefaults(
  defineProps<{
    workers: EditableWorkerSpec[]
    title?: string
    compact?: boolean
    showEnabled?: boolean
    rowActions?: boolean
    readOnly?: boolean
    error?: string | null
  }>(),
  {
    title: 'Registered Workers',
    compact: false,
    showEnabled: false,
    rowActions: false,
    readOnly: false,
    error: null,
  },
)

const emit = defineEmits<{
  save: [workers: EditableWorkerSpec[]]
  saveOne: [worker: EditableWorkerSpec, originalName: string | null]
  removeOne: [workerName: string]
}>()

const draft = ref<WorkerDraft[]>([])
let nextDraftId = 0

function nextLocalId(): string {
  nextDraftId += 1
  return `worker-draft-${nextDraftId}`
}

function toDraft(workers: EditableWorkerSpec[], previous: WorkerDraft[] = []): WorkerDraft[] {
  const previousByName = new Map(
    previous
      .filter((worker) => worker.persistedName)
      .map((worker) => [worker.persistedName, worker]),
  )

  return workers.map((worker) => ({
    localId: previousByName.get(worker.name)?.localId ?? nextLocalId(),
    persistedName: worker.name,
    name: worker.name,
    backend: worker.backend,
    editing: previousByName.get(worker.name)?.editing ?? !props.rowActions,
    enabled: worker.enabled !== false,
    runtimeId: worker.runtimeId ?? '',
    executorUrl: typeof worker.config?.url === 'string' ? worker.config.url : '',
    transport: editorWorkerTransport(worker.config?.transport),
    strataUrl: typeof worker.config?.strata_url === 'string' ? worker.config.strata_url : '',
    extraConfig:
      worker.config && typeof worker.config === 'object'
        ? Object.fromEntries(
            Object.entries(worker.config).filter(
              ([key]) => !['url', 'transport', 'strata_url'].includes(key),
            ),
          )
        : {},
  }))
}

watch(
  () => props.workers,
  (workers) => {
    draft.value = toDraft(workers, draft.value)
  },
  { immediate: true, deep: true },
)

function addWorker() {
  draft.value.push({
    localId: nextLocalId(),
    persistedName: '',
    name: '',
    backend: 'executor',
    editing: true,
    enabled: true,
    runtimeId: '',
    executorUrl: '',
    transport: 'direct',
    strataUrl: '',
    extraConfig: {},
  })
}

function serializeWorker(worker: WorkerDraft): EditableWorkerSpec {
  const config =
    worker.backend === 'executor'
      ? {
          ...worker.extraConfig,
          ...(worker.executorUrl.trim() ? { url: worker.executorUrl.trim() } : {}),
          ...(worker.transport === 'signed' ? { transport: 'signed' } : {}),
          ...(worker.transport === 'signed' && worker.strataUrl.trim()
            ? { strata_url: worker.strataUrl.trim() }
            : {}),
        }
      : {}

  return {
    name: worker.name.trim(),
    backend: worker.backend,
    ...(props.showEnabled ? { enabled: worker.enabled } : {}),
    runtimeId: worker.runtimeId.trim() || null,
    config,
  }
}

function removeWorker(index: number) {
  const worker = draft.value[index]
  if (props.rowActions && worker.persistedName) {
    emit('removeOne', worker.persistedName)
    return
  }
  draft.value.splice(index, 1)
}

function save() {
  emit(
    'save',
    draft.value.map(serializeWorker).filter((worker) => worker.name),
  )
}

function saveOne(index: number) {
  const worker = draft.value[index]
  const serialized = serializeWorker(worker)
  if (!serialized.name) return
  const originalName = worker.persistedName || null
  worker.persistedName = serialized.name
  worker.editing = false
  emit('saveOne', serialized, originalName)
}

function editWorker(index: number) {
  draft.value[index].editing = true
}

function cancelEdit(index: number) {
  const worker = draft.value[index]
  if (!worker.persistedName) {
    draft.value.splice(index, 1)
    return
  }
  const persisted = props.workers.find((candidate) => candidate.name === worker.persistedName)
  if (!persisted) {
    draft.value.splice(index, 1)
    return
  }
  draft.value[index] = toDraft([persisted], [worker])[0]
}
</script>

<template>
  <div class="worker-list-editor" :class="{ compact: compact, 'show-enabled': showEnabled }">
    <div class="worker-list-header">
      <span class="worker-list-title">{{ title }}</span>
      <button v-if="!readOnly" class="worker-list-add" @click="addWorker">+ Worker</button>
    </div>

    <div v-if="draft.length === 0" class="worker-list-empty">
      {{ readOnly ? 'No notebook workers configured' : 'No workers configured yet' }}
    </div>

    <div v-for="(worker, index) in draft" :key="worker.localId" class="worker-list-entry">
      <div
        v-if="rowActions && !worker.editing"
        class="worker-list-summary"
        :class="{ 'show-enabled': showEnabled }"
      >
        <div class="worker-summary-main">
          <label v-if="showEnabled" class="worker-enabled-toggle">
            <input v-model="worker.enabled" type="checkbox" :disabled="true" />
            <span>{{ worker.enabled ? 'enabled' : 'disabled' }}</span>
          </label>
          <code class="worker-summary-name">{{ worker.name || 'unnamed worker' }}</code>
          <span class="worker-summary-meta">{{ worker.backend }}</span>
          <span class="worker-summary-meta">{{
            worker.backend === 'executor' ? worker.transport : 'local'
          }}</span>
          <span v-if="worker.runtimeId" class="worker-summary-meta">{{ worker.runtimeId }}</span>
        </div>
        <div class="worker-row-actions">
          <button class="worker-list-save-row" :disabled="readOnly" @click="editWorker(index)">
            Edit
          </button>
          <button class="worker-list-remove" :disabled="readOnly" @click="removeWorker(index)">
            Delete
          </button>
        </div>
      </div>

      <template v-else>
        <div class="worker-list-row">
          <label v-if="showEnabled" class="worker-enabled-toggle">
            <input v-model="worker.enabled" type="checkbox" :disabled="readOnly" />
            <span>{{ worker.enabled ? 'enabled' : 'disabled' }}</span>
          </label>
          <input
            v-model="worker.name"
            class="worker-input"
            type="text"
            placeholder="name"
            :disabled="readOnly"
          />
          <select v-model="worker.backend" class="worker-input" :disabled="readOnly">
            <option value="local">local</option>
            <option value="executor">executor</option>
          </select>
          <input
            v-model="worker.runtimeId"
            class="worker-input"
            type="text"
            placeholder="runtime id (optional)"
            :disabled="readOnly"
          />
          <input
            v-model="worker.executorUrl"
            class="worker-input"
            type="text"
            placeholder="executor url (executor only)"
            :disabled="readOnly"
          />
          <select
            v-model="worker.transport"
            class="worker-input"
            :disabled="readOnly || worker.backend !== 'executor'"
          >
            <option value="direct">direct</option>
            <option value="signed">signed</option>
          </select>
          <input
            v-model="worker.strataUrl"
            class="worker-input"
            type="text"
            placeholder="strata url (signed only, optional)"
            :disabled="readOnly || worker.backend !== 'executor' || worker.transport !== 'signed'"
          />
          <div class="worker-row-actions">
            <template v-if="rowActions">
              <button class="worker-list-save-row" :disabled="readOnly" @click="saveOne(index)">
                Save
              </button>
              <button class="worker-list-remove" :disabled="readOnly" @click="cancelEdit(index)">
                {{ worker.persistedName ? 'Cancel' : '×' }}
              </button>
            </template>
            <button v-else-if="!readOnly" class="worker-list-remove" @click="removeWorker(index)">
              ×
            </button>
          </div>
        </div>

        <div class="worker-list-hint">
          <template v-if="worker.backend === 'executor' && worker.transport === 'signed'">
            Signed transport uses the Strata build + signed-URL path. Set <code>strata_url</code>
            when the worker cannot reach the default server URL.
          </template>
          <template v-else-if="worker.backend === 'executor'">
            Direct transport uploads notebook inputs to the executor over HTTP and returns the
            output bundle directly.
          </template>
          <template v-else>
            Local workers execute in the notebook runtime on this machine.
          </template>
        </div>
      </template>
    </div>

    <div v-if="!readOnly && !rowActions" class="worker-list-actions">
      <button class="worker-list-save" @click="save">Save workers</button>
    </div>

    <div v-if="error" class="worker-list-error">
      {{ error }}
    </div>
  </div>
</template>

<style scoped>
.worker-list-editor {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.worker-list-editor.compact {
  gap: 6px;
}

.worker-list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.worker-list-title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-secondary);
}

.worker-list-add,
.worker-list-save,
.worker-list-save-row,
.worker-list-remove {
  background: var(--bg-input);
  border: 1px solid var(--border-strong);
  color: var(--text-primary);
  border-radius: 6px;
  cursor: pointer;
}

.worker-list-add,
.worker-list-save {
  padding: 6px 10px;
  font-size: 12px;
}

.worker-list-save-row {
  padding: 6px 10px;
  font-size: 12px;
}

.worker-list-remove {
  min-width: 28px;
  height: 30px;
  padding: 0 10px;
  font-size: 14px;
}

.worker-row-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  justify-content: flex-end;
  grid-column: 1 / -1;
  flex-wrap: wrap;
}

.worker-list-entry {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.worker-list-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 6px;
  align-items: start;
}

.worker-list-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid var(--bg-input);
  border-radius: 8px;
  background: var(--bg-surface);
}

.worker-summary-main {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  flex-wrap: wrap;
}

.worker-summary-name {
  color: var(--text-primary);
  font-size: 12px;
}

.worker-summary-meta {
  color: var(--text-secondary);
  font-size: 12px;
}

.worker-enabled-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  border-radius: 6px;
  border: 1px solid var(--bg-input);
  background: var(--bg-surface);
  color: var(--cat-subtext1);
  font-size: 12px;
}

.worker-enabled-toggle input {
  margin: 0;
}

.worker-input {
  width: 100%;
  min-width: 0;
  padding: 6px 8px;
  background: var(--bg-base);
  border: 1px solid var(--bg-input);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 12px;
}

.worker-list-empty {
  color: var(--text-muted);
  font-size: 12px;
}

.worker-list-hint {
  margin-top: -2px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.4;
}

.worker-list-hint code {
  color: var(--accent-primary);
}

.worker-list-actions {
  display: flex;
  justify-content: flex-end;
}

.worker-list-error {
  color: var(--accent-danger);
  font-size: 12px;
  line-height: 1.4;
}

@media (max-width: 920px) {
  .worker-list-row {
    grid-template-columns: 1fr;
  }
}
</style>
