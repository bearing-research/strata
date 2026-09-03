<script setup lang="ts">
import { onMounted } from 'vue'
import { useNotebook } from '../stores/notebook'

const { profilingSummary, requestProfilingSummary } = useNotebook()

onMounted(() => {
  requestProfilingSummary()
})

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function refresh() {
  requestProfilingSummary()
}
</script>

<template>
  <div class="profiling-panel">
    <div class="profiling-header">
      <span>Profiling</span>
      <button class="refresh-btn" title="Refresh" @click="refresh">&#x21BB;</button>
    </div>

    <div v-if="profilingSummary" class="profiling-body">
      <div class="profiling-stats">
        <div class="stat-row">
          <span class="stat-label">Total execution</span>
          <span class="stat-value">{{ formatMs(profilingSummary.totalExecutionMs) }}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Cache savings</span>
          <span class="stat-value cache-savings">
            ~{{ formatMs(profilingSummary.cacheSavingsMs) }}
            <span class="stat-detail">({{ profilingSummary.cacheHits }} hits)</span>
          </span>
        </div>
        <!--
          Only shown once a team hit has actually happened. A permanent "0 from
          your team" row on a solo notebook advertises a feature the reader has
          not turned on, which is noise rather than information.
        -->
        <div v-if="profilingSummary.teamCacheHits > 0" class="stat-row">
          <span class="stat-label">From your team</span>
          <span class="stat-value cache-savings">
            ~{{ formatMs(profilingSummary.teamCacheSavingsMs) }}
            <span class="stat-detail">
              ({{ profilingSummary.teamCacheHits }}
              {{ profilingSummary.teamCacheHits === 1 ? 'hit' : 'hits'
              }}{{
                profilingSummary.teamContributors.length
                  ? ', ' + profilingSummary.teamContributors.join(', ')
                  : ''
              }})
            </span>
          </span>
        </div>
        <div v-if="profilingSummary.totalArtifactBytes > 0" class="stat-row">
          <span class="stat-label">Artifact storage</span>
          <span class="stat-value">{{ formatBytes(profilingSummary.totalArtifactBytes) }}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Cells</span>
          <span class="stat-value">
            {{ profilingSummary.cacheHits }} cached, {{ profilingSummary.cacheMisses }} computed
          </span>
        </div>
      </div>

      <div v-if="profilingSummary.cellProfiles.length" class="cell-profiles">
        <div v-for="cp in profilingSummary.cellProfiles" :key="cp.cellId" class="cell-profile-row">
          <span class="cp-name" :title="cp.cellId">{{ cp.cellName }}</span>
          <span class="cp-status" :class="`cp-${cp.status}`">
            {{ cp.cacheHit ? '&#x26A1;' : '&#x25D0;' }}
          </span>
          <span class="cp-duration">{{ cp.durationMs ? formatMs(cp.durationMs) : '-' }}</span>
        </div>
      </div>
    </div>
    <div v-else class="profiling-empty">No profiling data yet. Run some cells first.</div>
  </div>
</template>

<style scoped>
.profiling-panel {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  margin-top: 8px;
  min-width: 180px;
  overflow: auto;
}
.profiling-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 1px solid var(--border-subtle);
}
.refresh-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 14px;
  padding: 0 4px;
  border-radius: 3px;
}
.refresh-btn:hover {
  color: var(--accent-primary);
}

.profiling-body {
  padding: 8px;
}
.profiling-stats {
  margin-bottom: 8px;
}
.stat-row {
  display: flex;
  justify-content: space-between;
  padding: 2px 4px;
  font-size: 11px;
}
.stat-label {
  color: var(--text-muted);
}
.stat-value {
  color: var(--text-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.cache-savings {
  color: var(--accent-success);
}
.stat-detail {
  color: var(--text-muted);
  font-size: 10px;
}

.cell-profiles {
  border-top: 1px solid var(--border-subtle);
  padding-top: 6px;
}
.cell-profile-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 2px 4px;
  font-size: 11px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.cp-name {
  flex: 1;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cp-status {
  font-size: 10px;
}
.cp-ready {
  color: var(--accent-success);
}
.cp-stale {
  color: var(--accent-warning);
}
.cp-error {
  color: var(--accent-danger);
}
.cp-idle {
  color: var(--text-muted);
}
.cp-duration {
  color: var(--text-muted);
  min-width: 36px;
  text-align: right;
}

.profiling-empty {
  padding: 12px;
  font-size: 11px;
  color: var(--text-muted);
  text-align: center;
}
</style>
