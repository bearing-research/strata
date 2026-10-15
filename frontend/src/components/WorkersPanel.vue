<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useNotebook } from '../stores/notebook'
import type { WorkerCatalogEntry, WorkerHealthHistoryEntry } from '../types/notebook'
import {
  summarizeWorkerCatalog,
  workerAttentionReason,
  workerNeedsAttention,
  workerTransportLabel,
} from '../utils/notebookWorkers'
import WorkerConfigEditor from './WorkerConfigEditor.vue'
import WorkerListEditor from './WorkerListEditor.vue'

const {
  notebook,
  connected,
  availableWorkers,
  workerDefinitionsEditable,
  serviceMode,
  serverManagedWorkers,
  serverWorkerRegistryAvailable,
  serverWorkerRegistryLoading,
  workerHealthLoading,
  workerHealthCheckedAt,
  notebookWorkerError,
  workerRegistryError,
  serverWorkerRegistryError,
  fetchWorkers,
  ensureWorkersLoaded,
  updateNotebookWorkerAction,
  updateNotebookWorkersAction,
  updateServerWorkerRegistryAction,
  saveServerWorkerAction,
  deleteServerWorkerAction,
  updateServerWorkerEnabledAction,
  refreshServerWorkerAction,
  isServerWorkerActionLoading,
} = useNotebook()
const showPanel = ref(false)

const workerLabel = computed(() => notebook.worker || 'local')
const registryManagedByServer = computed(() => !workerDefinitionsEditable.value)
const canEditServerRegistry = computed(
  () => registryManagedByServer.value && serverWorkerRegistryAvailable.value,
)
const workerModeLabel = computed(() => (serviceMode.value ? 'Service mode' : 'Personal mode'))
const workerModeTitle = computed(() =>
  registryManagedByServer.value
    ? 'This notebook selects from a shared server-managed worker catalog.'
    : 'This notebook owns its worker catalog and stores it beside the notebook.',
)
const workerModeDescription = computed(() => {
  if (!registryManagedByServer.value) {
    return 'Use this mode for local notebook work, ad hoc executor URLs, and notebook-specific worker definitions.'
  }

  if (canEditServerRegistry.value) {
    return 'This browser has notebook-worker admin access, so it can edit the shared catalog as well as choose a notebook default.'
  }

  return 'This notebook can choose from the shared catalog below, but only an admin can change worker definitions or enable and disable workers.'
})
const lastCheckedLabel = computed(() => {
  if (!workerHealthCheckedAt.value) {
    return 'Not checked yet'
  }

  return new Date(workerHealthCheckedAt.value).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
  })
})

function workerCheckedLabel(rawCheckedAt: number | null | undefined): string {
  if (!rawCheckedAt) return 'not checked'
  return new Date(rawCheckedAt).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
  })
}

function workerProbeLabel(count: number | null | undefined): string {
  if (!count) return 'no probes yet'
  return count === 1 ? '1 probe' : `${count} probes`
}

function workerDurationLabel(rawDuration: number | null | undefined): string {
  if (typeof rawDuration !== 'number' || !Number.isFinite(rawDuration)) {
    return 'latency unknown'
  }
  return `${rawDuration} ms`
}

function workerHistoryLabel(entry: WorkerHealthHistoryEntry): string {
  const duration = workerDurationLabel(entry.durationMs)
  return `${workerCheckedLabel(entry.checkedAt)} ${entry.health} · ${duration}`
}

function workerHistoryTitle(entry: WorkerHealthHistoryEntry): string {
  return entry.error ? `${workerHistoryLabel(entry)}\n${entry.error}` : workerHistoryLabel(entry)
}

const workerSummary = computed(() => summarizeWorkerCatalog(availableWorkers.value))

function workerAttentionSeverity(worker: WorkerCatalogEntry): number {
  if (worker.enabled === false || worker.allowed === false) return 4
  if (worker.health === 'unavailable') return 3
  if ((worker.consecutiveFailures ?? 0) > 0) return 2
  if (worker.lastError) return 1
  return 0
}

const attentionWorkers = computed(() =>
  [...availableWorkers.value]
    .filter((worker) => workerNeedsAttention(worker))
    .sort((left, right) => {
      const severityDelta = workerAttentionSeverity(right) - workerAttentionSeverity(left)
      if (severityDelta !== 0) return severityDelta
      return left.name.localeCompare(right.name)
    }),
)

function workerAttentionLabel(worker: WorkerCatalogEntry): string {
  return workerAttentionReason(worker) || 'Needs attention'
}

watch(showPanel, (open) => {
  if (open) {
    void ensureWorkersLoaded()
  }
})
</script>

<template>
  <div class="workers-panel">
    <button class="workers-toggle" @click="showPanel = !showPanel">
      Worker
      <span class="worker-label">{{ workerLabel }}</span>
      <span class="toggle-icon">{{ showPanel ? '&#9650;' : '&#9660;' }}</span>
    </button>

    <div v-if="showPanel" class="workers-content">
      <div class="workers-health-row">
        <span class="workers-health-text"> Last checked: {{ lastCheckedLabel }} </span>
        <button
          class="workers-refresh"
          :disabled="!connected || workerHealthLoading"
          @click="fetchWorkers(true)"
        >
          {{ workerHealthLoading ? 'Refreshing…' : 'Refresh health' }}
        </button>
      </div>

      <div v-if="availableWorkers.length" class="workers-summary-grid">
        <div class="workers-summary-card">
          <span class="workers-summary-label">Visible</span>
          <strong class="workers-summary-value">{{ workerSummary.total }}</strong>
          <span class="workers-summary-detail">workers in this notebook view</span>
        </div>
        <div class="workers-summary-card">
          <span class="workers-summary-label">Healthy</span>
          <strong class="workers-summary-value">{{ workerSummary.healthy }}</strong>
          <span class="workers-summary-detail">
            {{ workerSummary.unavailable }} unavailable · {{ workerSummary.unknown }} unknown
          </span>
        </div>
        <div class="workers-summary-card" :class="{ attention: workerSummary.attention > 0 }">
          <span class="workers-summary-label">Attention</span>
          <strong class="workers-summary-value">{{ workerSummary.attention }}</strong>
          <span class="workers-summary-detail">
            {{ workerSummary.disabled }} disabled · {{ workerSummary.blocked }} blocked
          </span>
        </div>
      </div>

      <div
        class="workers-mode-card"
        :class="{
          service: registryManagedByServer,
          personal: !registryManagedByServer,
        }"
      >
        <div class="workers-mode-header">
          <span class="workers-mode-eyebrow">Worker policy</span>
          <span class="workers-mode-badge">{{ workerModeLabel }}</span>
        </div>
        <p class="workers-mode-title">{{ workerModeTitle }}</p>
        <p class="workers-copy">{{ workerModeDescription }}</p>
        <p class="workers-copy workers-copy-muted">
          Notebook default applies to new runs, and individual cells can still override it.
        </p>
      </div>

      <WorkerConfigEditor
        :worker="notebook.worker"
        :options="availableWorkers"
        title="Notebook Default Worker"
        :read-only="!connected"
        :error="notebookWorkerError"
        @save="updateNotebookWorkerAction"
      />

      <WorkerListEditor
        v-if="workerDefinitionsEditable"
        :workers="notebook.workers"
        title="Notebook Worker Catalog"
        :read-only="!connected || !workerDefinitionsEditable"
        :error="workerRegistryError"
        @save="updateNotebookWorkersAction"
      />
      <WorkerListEditor
        v-else-if="canEditServerRegistry"
        :workers="serverManagedWorkers"
        title="Server Worker Catalog"
        :show-enabled="true"
        :row-actions="true"
        :read-only="!connected || serverWorkerRegistryLoading"
        :error="serverWorkerRegistryError"
        @save="updateServerWorkerRegistryAction"
        @save-one="saveServerWorkerAction"
        @remove-one="deleteServerWorkerAction"
      />
      <div v-else class="workers-copy workers-copy-muted">
        You are viewing the shared server catalog. Notebook runs can pick from these workers, but
        registry edits require notebook-worker admin access.
      </div>

      <div v-if="availableWorkers.length" class="workers-catalog">
        <div v-if="attentionWorkers.length" class="workers-attention">
          <div class="workers-catalog-title">Attention Needed</div>
          <div
            v-for="worker in attentionWorkers"
            :key="`attention-${worker.name}`"
            class="workers-attention-card"
          >
            <div class="workers-attention-main">
              <div class="workers-attention-row">
                <code>{{ worker.name }}</code>
                <span class="workers-catalog-meta">{{ worker.backend }}</span>
                <span class="workers-catalog-meta">{{
                  worker.transport || workerTransportLabel(worker)
                }}</span>
                <span
                  class="workers-catalog-meta"
                  :class="{
                    unhealthy: worker.health === 'unavailable',
                    unknown: worker.health === 'unknown',
                    warming: worker.health === 'warming',
                  }"
                >
                  {{ worker.health }}
                </span>
              </div>
              <div class="workers-attention-reason">
                {{ workerAttentionLabel(worker) }}
              </div>
            </div>
            <div
              v-if="canEditServerRegistry && worker.source === 'server'"
              class="workers-catalog-actions"
            >
              <button
                class="workers-action"
                :disabled="!connected || isServerWorkerActionLoading(worker.name)"
                @click="refreshServerWorkerAction(worker.name)"
              >
                {{ isServerWorkerActionLoading(worker.name) ? 'Refreshing…' : 'Refresh' }}
              </button>
              <button
                class="workers-action"
                :disabled="!connected || isServerWorkerActionLoading(worker.name)"
                @click="updateServerWorkerEnabledAction(worker.name, worker.enabled === false)"
              >
                {{
                  isServerWorkerActionLoading(worker.name)
                    ? 'Saving…'
                    : worker.enabled === false
                      ? 'Enable'
                      : 'Disable'
                }}
              </button>
            </div>
          </div>
        </div>

        <div class="workers-catalog-title">Visible Workers</div>
        <div v-for="worker in availableWorkers" :key="worker.name" class="workers-catalog-card">
          <div class="workers-catalog-row">
            <code>{{ worker.name }}</code>
            <span class="workers-catalog-meta">{{ worker.source || 'unknown' }}</span>
            <span class="workers-catalog-meta">{{ worker.backend }}</span>
            <span class="workers-catalog-meta">{{
              worker.transport || workerTransportLabel(worker)
            }}</span>
            <span
              class="workers-catalog-meta"
              :class="{
                unhealthy: worker.health === 'unavailable',
                unknown: worker.health === 'unknown',
              }"
            >
              {{ worker.health }}
            </span>
            <span
              class="workers-catalog-meta"
              :class="{
                disallowed: worker.allowed === false,
                disabled: worker.enabled === false,
              }"
            >
              {{
                worker.enabled === false
                  ? 'disabled'
                  : worker.allowed === false
                    ? 'not allowed'
                    : 'allowed'
              }}
            </span>
            <div
              v-if="canEditServerRegistry && worker.source === 'server'"
              class="workers-catalog-actions"
            >
              <button
                class="workers-action"
                :disabled="!connected || isServerWorkerActionLoading(worker.name)"
                @click="refreshServerWorkerAction(worker.name)"
              >
                {{ isServerWorkerActionLoading(worker.name) ? 'Refreshing…' : 'Refresh' }}
              </button>
              <button
                class="workers-action"
                :disabled="!connected || isServerWorkerActionLoading(worker.name)"
                @click="updateServerWorkerEnabledAction(worker.name, worker.enabled === false)"
              >
                {{
                  isServerWorkerActionLoading(worker.name)
                    ? 'Saving…'
                    : worker.enabled === false
                      ? 'Enable'
                      : 'Disable'
                }}
              </button>
            </div>
          </div>
          <div class="workers-catalog-detail-row">
            <span class="workers-catalog-detail">
              last checked {{ workerCheckedLabel(worker.healthCheckedAt) }}
            </span>
            <span class="workers-catalog-detail">
              {{ workerProbeLabel(worker.probeCount) }}
            </span>
            <span v-if="worker.consecutiveFailures" class="workers-catalog-detail unhealthy">
              {{ worker.consecutiveFailures }} consecutive failures
            </span>
            <span v-if="worker.lastProbeDurationMs != null" class="workers-catalog-detail">
              last probe {{ workerDurationLabel(worker.lastProbeDurationMs) }}
            </span>
            <span v-if="worker.healthUrl" class="workers-catalog-detail">
              health {{ worker.healthUrl }}
            </span>
          </div>
          <div class="workers-catalog-detail-row">
            <span v-if="worker.lastHealthyAt" class="workers-catalog-detail">
              last healthy {{ workerCheckedLabel(worker.lastHealthyAt) }}
            </span>
            <span v-if="worker.lastUnavailableAt" class="workers-catalog-detail">
              last unavailable {{ workerCheckedLabel(worker.lastUnavailableAt) }}
            </span>
            <span v-if="worker.lastStatusChangeAt" class="workers-catalog-detail">
              status changed {{ workerCheckedLabel(worker.lastStatusChangeAt) }}
            </span>
          </div>
          <div
            v-if="worker.healthHistory && worker.healthHistory.length"
            class="workers-catalog-history"
          >
            <span class="workers-catalog-history-label">recent probes</span>
            <span
              v-for="entry in worker.healthHistory.slice(0, 4)"
              :key="`${worker.name}-${entry.checkedAt}-${entry.health}`"
              class="workers-catalog-history-chip"
              :class="{
                unhealthy: entry.health === 'unavailable',
                unknown: entry.health === 'unknown',
              }"
              :title="workerHistoryTitle(entry)"
            >
              {{ workerHistoryLabel(entry) }}
            </span>
          </div>
          <div v-if="worker.lastError" class="workers-catalog-error">
            {{ worker.lastError }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.workers-panel {
  margin-top: 12px;
  border-top: 1px solid var(--border-subtle);
  padding-top: 8px;
}

.workers-toggle {
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

.workers-toggle:hover {
  color: var(--text-primary);
}

.worker-label {
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

.workers-content {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.workers-health-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.workers-health-text {
  font-size: 12px;
  color: var(--text-muted);
}

.workers-refresh {
  border: 1px solid var(--bg-input);
  background: var(--bg-surface);
  color: var(--text-primary);
  border-radius: 8px;
  padding: 5px 10px;
  font-size: 12px;
  cursor: pointer;
}

.workers-refresh:disabled {
  opacity: 0.55;
  cursor: default;
}

.workers-copy {
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.4;
}

.workers-copy-muted {
  margin-top: -2px;
}

.workers-mode-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border: 1px solid var(--bg-input);
  border-radius: 12px;
  background: var(--bg-base);
}

.workers-mode-card.service {
  border-color: var(--tint-primary-strong);
  background: var(--tint-primary);
}

.workers-mode-card.personal {
  border-color: var(--tint-success);
  background: var(--tint-success);
}

.workers-mode-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.workers-mode-eyebrow {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.workers-mode-badge {
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  color: var(--text-primary);
  background: var(--bg-input);
}

.workers-mode-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  line-height: 1.4;
}

.workers-summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 8px;
}

.workers-summary-card {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 10px;
  border: 1px solid var(--bg-input);
  border-radius: 10px;
  background: var(--bg-base);
}

.workers-summary-card.attention {
  border-color: var(--tint-warning);
  background: var(--tint-warning);
}

.workers-summary-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.workers-summary-value {
  font-size: 18px;
  line-height: 1;
  color: var(--text-primary);
}

.workers-summary-detail {
  font-size: 11px;
  color: var(--text-muted);
}

.workers-catalog {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.workers-attention {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.workers-attention-card {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 10px;
  border: 1px solid var(--tint-warning);
  border-radius: 10px;
  background: var(--tint-warning);
}

.workers-attention-main {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.workers-attention-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--cat-subtext1);
}

.workers-attention-reason {
  font-size: 12px;
  color: var(--accent-warning);
  line-height: 1.4;
}

.workers-catalog-card {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 10px;
  border: 1px solid var(--bg-input);
  border-radius: 10px;
  background: var(--bg-base);
}

.workers-catalog-title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-secondary);
}

.workers-catalog-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  font-size: 12px;
  color: var(--cat-subtext1);
}

.workers-catalog-meta {
  padding: 1px 6px;
  border-radius: 999px;
  background: var(--bg-input);
  color: var(--accent-primary);
  font-size: 11px;
}

.workers-catalog-meta.disallowed {
  color: var(--accent-danger);
}

.workers-catalog-meta.disabled,
.workers-catalog-meta.unknown {
  color: var(--accent-peach);
}

.workers-catalog-meta.unhealthy {
  color: var(--accent-danger);
}

.workers-catalog-meta.warming {
  color: var(--accent-primary);
  font-style: italic;
}

.workers-catalog-actions {
  margin-left: auto;
  display: flex;
  gap: 6px;
}

.workers-action {
  border: 1px solid var(--bg-input);
  background: var(--bg-surface);
  color: var(--text-primary);
  border-radius: 8px;
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
}

.workers-action:disabled {
  opacity: 0.55;
  cursor: default;
}

.workers-catalog-detail-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.workers-catalog-detail {
  font-size: 11px;
  color: var(--text-muted);
}

.workers-catalog-error {
  font-size: 12px;
  color: var(--accent-danger);
  line-height: 1.4;
}

.workers-catalog-history {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.workers-catalog-history-label {
  font-size: 11px;
  color: var(--text-muted);
}

.workers-catalog-history-chip {
  padding: 1px 6px;
  border-radius: 999px;
  background: var(--bg-surface);
  border: 1px solid var(--bg-input);
  color: var(--text-secondary);
  font-size: 10px;
}

.workers-catalog-history-chip.unhealthy {
  color: var(--accent-danger);
}

.workers-catalog-history-chip.unknown {
  color: var(--accent-peach);
}

@media (max-width: 900px) {
  .workers-attention-card {
    flex-direction: column;
  }

  .workers-catalog-actions {
    margin-left: 0;
  }
}
</style>
