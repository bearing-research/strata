<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'

// A plain back control: return to wherever the user came from (usually the
// notebook they opened Artifacts from), falling back to the home list.
const router = useRouter()
function goBack() {
  if (window.history.length > 1) router.back()
  else router.push('/')
}
import {
  useStrata,
  type ArtifactQuery,
  type ArtifactRow,
  type ArtifactStats,
} from '../composables/useStrata'
import ThemeToggle from '../components/ThemeToggle.vue'

const strata = useStrata()

const PAGE_SIZE = 100
const STATES = ['', 'ready', 'building', 'failed'] as const
type SortKey = NonNullable<ArtifactQuery['sort']>

const stats = ref<ArtifactStats | null>(null)
const rows = ref<ArtifactRow[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

const stateFilter = ref('')
const namePrefix = ref('')
const sort = ref<SortKey>('created_at')
const order = ref<'asc' | 'desc'>('desc')
const offset = ref(0)

function formatBytes(bytes: number | null): string {
  if (bytes === null || bytes === undefined) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let i = 0
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024
    i += 1
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[i]}`
}

function formatRows(count: number | null): string {
  if (count === null || count === undefined) return '—'
  return count.toLocaleString()
}

function formatTime(ts: number | null): string {
  if (ts === null || ts === undefined) return '—'
  // created_at is epoch seconds.
  return new Date(ts * 1000).toLocaleString()
}

async function loadStats() {
  try {
    stats.value = await strata.getArtifactStats()
  } catch {
    // Stats failing shouldn't blank the table; the list load surfaces the
    // error banner if the store is truly unreachable.
    stats.value = null
  }
}

async function loadArtifacts() {
  loading.value = true
  error.value = null
  try {
    const data = await strata.getArtifacts({
      limit: PAGE_SIZE,
      offset: offset.value,
      state: stateFilter.value || undefined,
      namePrefix: namePrefix.value.trim() || undefined,
      sort: sort.value,
      order: order.value,
    })
    rows.value = data.artifacts
  } catch (e: any) {
    error.value = e?.message || 'Failed to fetch artifacts'
    rows.value = []
  } finally {
    loading.value = false
  }
}

function refresh() {
  void loadStats()
  void loadArtifacts()
}

// Clicking a sortable header sets that column, toggling asc/desc when it's
// already the active sort. Any sort/filter change returns to page 1.
function setSort(key: SortKey) {
  if (sort.value === key) {
    order.value = order.value === 'asc' ? 'desc' : 'asc'
  } else {
    sort.value = key
    order.value = 'desc'
  }
}

function sortIndicator(key: SortKey): string {
  if (sort.value !== key) return ''
  return order.value === 'asc' ? '▲' : '▼'
}

function shortId(id: string): string {
  return id.length > 20 ? `${id.slice(0, 17)}…` : id
}

const canPrev = computed(() => offset.value > 0)
const canNext = computed(() => rows.value.length === PAGE_SIZE)

function prevPage() {
  if (!canPrev.value) return
  offset.value = Math.max(0, offset.value - PAGE_SIZE)
}

function nextPage() {
  if (!canNext.value) return
  offset.value += PAGE_SIZE
}

// Filters/sort reset to page 1; the offset watcher then refetches. When
// already on page 1 the offset watcher won't fire, so refetch directly.
watch([stateFilter, namePrefix, sort, order], () => {
  if (offset.value !== 0) offset.value = 0
  else void loadArtifacts()
})

watch(offset, () => {
  void loadArtifacts()
})

onMounted(() => {
  refresh()
})
</script>

<template>
  <div class="artifacts" data-testid="artifacts-page">
    <div class="artifacts-header">
      <div class="artifacts-title">
        <button type="button" class="nav-link back-btn" data-testid="page-back" @click="goBack">
          ← Back
        </button>
        <router-link to="/" class="back-link" title="Back to notebooks">◆ strata</router-link>
        <span class="crumb">/</span>
        <span class="page-name">artifacts</span>
      </div>
      <div class="artifacts-header-actions">
        <router-link to="/logs" class="nav-link" data-testid="nav-logs">Logs</router-link>
        <ThemeToggle />
      </div>
    </div>

    <div v-if="stats" class="stat-strip" data-testid="artifact-stats">
      <div class="stat-card">
        <span class="stat-value">{{ stats.total_versions.toLocaleString() }}</span>
        <span class="stat-label">versions</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{{ stats.name_count.toLocaleString() }}</span>
        <span class="stat-label">names</span>
      </div>
      <div class="stat-card">
        <span class="stat-value ready">{{ stats.ready_versions.toLocaleString() }}</span>
        <span class="stat-label">ready</span>
      </div>
      <div class="stat-card">
        <span class="stat-value building">{{ stats.building_versions.toLocaleString() }}</span>
        <span class="stat-label">building</span>
      </div>
      <div class="stat-card">
        <span class="stat-value failed">{{ stats.failed_versions.toLocaleString() }}</span>
        <span class="stat-label">failed</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{{ formatBytes(stats.total_bytes) }}</span>
        <span class="stat-label">total size</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{{ stats.total_rows.toLocaleString() }}</span>
        <span class="stat-label">total rows</span>
      </div>
    </div>

    <div class="filter-bar">
      <label class="filter">
        State
        <select v-model="stateFilter" class="control" data-testid="artifacts-state">
          <option v-for="s in STATES" :key="s" :value="s">{{ s ? s : 'all' }}</option>
        </select>
      </label>
      <label class="filter filter-grow">
        Name prefix
        <input
          v-model="namePrefix"
          class="control"
          type="text"
          placeholder="filter by name prefix"
          data-testid="artifacts-name-prefix"
        />
      </label>
      <button class="btn" :disabled="loading" data-testid="artifacts-refresh" @click="refresh">
        Refresh
      </button>
    </div>

    <div v-if="error" class="banner banner-error" data-testid="artifacts-error">{{ error }}</div>

    <div class="table-wrap">
      <table class="artifact-table" data-testid="artifacts-table">
        <thead>
          <tr>
            <th>Artifact ID</th>
            <th>Ver</th>
            <th>State</th>
            <th class="sortable num" @click="setSort('row_count')">
              Rows <span class="sort-ind">{{ sortIndicator('row_count') }}</span>
            </th>
            <th class="sortable num" @click="setSort('byte_size')">
              Size <span class="sort-ind">{{ sortIndicator('byte_size') }}</span>
            </th>
            <th class="sortable" @click="setSort('created_at')">
              Created <span class="sort-ind">{{ sortIndicator('created_at') }}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-if="!rows.length && !loading">
            <td colspan="6" class="empty">No artifacts match these filters.</td>
          </tr>
          <tr
            v-for="a in rows"
            :key="a.artifact_uri"
            :data-testid="`artifact-row-${a.artifact_id}-${a.version}`"
          >
            <td class="mono" :title="a.artifact_id">{{ shortId(a.artifact_id) }}</td>
            <td class="num">{{ a.version }}</td>
            <td>
              <span class="state-pill" :class="`state-${a.state}`">{{ a.state }}</span>
            </td>
            <td class="num">{{ formatRows(a.row_count) }}</td>
            <td class="num">{{ formatBytes(a.byte_size) }}</td>
            <td>{{ formatTime(a.created_at) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="pager">
      <button
        class="btn btn-secondary"
        :disabled="!canPrev"
        data-testid="artifacts-prev"
        @click="prevPage"
      >
        ‹ Prev
      </button>
      <span class="page-info">rows {{ offset + 1 }}–{{ offset + rows.length }}</span>
      <button
        class="btn btn-secondary"
        :disabled="!canNext"
        data-testid="artifacts-next"
        @click="nextPage"
      >
        Next ›
      </button>
    </div>
  </div>
</template>

<style scoped>
.artifacts {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 16px 24px;
  gap: 12px;
}

.artifacts-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.artifacts-title {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.artifacts-header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.back-link {
  font-weight: 700;
  font-size: 20px;
  color: var(--accent-primary);
  text-decoration: none;
}

.crumb {
  color: var(--text-muted);
}

.page-name {
  font-size: 20px;
  font-weight: 300;
  color: var(--text-muted);
}

.nav-link {
  font-size: 13px;
  color: var(--text-secondary);
  text-decoration: none;
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
}
/* .nav-link on a <button> (Back): reset the browser's default chrome. */
.back-btn {
  font-family: inherit;
  line-height: 1.4;
  background: transparent;
  cursor: pointer;
}

.nav-link:hover {
  color: var(--text-primary);
  border-color: var(--border-strong);
}

.stat-strip {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.stat-card {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 16px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  min-width: 90px;
}

.stat-value {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
}

.stat-value.ready {
  color: var(--accent-primary);
}

.stat-value.building {
  color: var(--accent-warning);
}

.stat-value.failed {
  color: var(--accent-danger);
}

.stat-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
}

.filter-bar {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  flex-wrap: wrap;
}

.filter {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 11px;
  color: var(--text-secondary);
}

.filter-grow {
  flex: 1;
  min-width: 180px;
}

.control {
  padding: 6px 10px;
  background: var(--bg-input);
  border: 1px solid var(--border-strong);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 13px;
}

.btn {
  padding: 7px 14px;
  border: 1px solid var(--accent-primary);
  background: var(--accent-primary);
  color: var(--bg-base, #fff);
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
}

.btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.btn-secondary {
  background: transparent;
  color: var(--text-secondary);
  border-color: var(--border-strong);
}

.banner {
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
}

.banner-error {
  background: var(--tint-danger);
  border: 1px solid var(--accent-danger);
  color: var(--accent-danger);
}

.table-wrap {
  flex: 1;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-surface);
}

.artifact-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.artifact-table th {
  position: sticky;
  top: 0;
  background: var(--bg-elevated);
  text-align: left;
  padding: 8px 12px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}

.artifact-table th.sortable {
  cursor: pointer;
  user-select: none;
}

.artifact-table th.sortable:hover {
  color: var(--text-primary);
}

.sort-ind {
  color: var(--accent-primary);
  font-size: 10px;
}

.artifact-table td {
  padding: 6px 12px;
  border-bottom: 1px solid var(--border);
  color: var(--text-primary);
}

.artifact-table tr:hover td {
  background: var(--bg-hover);
}

.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.mono {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  color: var(--text-secondary);
}

.empty {
  text-align: center;
  color: var(--text-muted);
  padding: 24px;
}

.state-pill {
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
}

.state-ready {
  background: var(--tint-primary);
  color: var(--accent-primary);
}

.state-building {
  background: var(--tint-warning, var(--bg-input));
  color: var(--accent-warning);
}

.state-failed {
  background: var(--tint-danger);
  color: var(--accent-danger);
}

.pager {
  display: flex;
  align-items: center;
  gap: 16px;
  justify-content: center;
}

.page-info {
  font-size: 12px;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}
</style>
