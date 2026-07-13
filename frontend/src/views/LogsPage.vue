<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useStrata, type LogEntry } from '../composables/useStrata'
import ThemeToggle from '../components/ThemeToggle.vue'

const strata = useStrata()

const LEVELS = ['', 'debug', 'info', 'warning', 'error'] as const
const MAX_ENTRIES = 2000
const POLL_INTERVAL_MS = 1000
const FETCH_LIMIT = 500

// Known fields rendered in the row itself; everything else is structured
// context shown in the expandable detail row.
const KNOWN_FIELDS = new Set(['cursor', 'timestamp', 'level', 'logger', 'message', 'notebook_id'])

const level = ref('')
const notebook = ref('')
const regex = ref('')
const regexError = ref<string | null>(null)
const liveTail = ref(false)
const loading = ref(false)
const error = ref<string | null>(null)

const entries = ref<LogEntry[]>([])
const cursor = ref(0)
const expanded = ref<Set<number>>(new Set())

const listEl = ref<HTMLElement | null>(null)
// When the user scrolls up we stop auto-following so reading history isn't
// yanked away by incoming lines; re-enabled once they scroll back to bottom.
const stickToBottom = ref(true)
let pollTimer: ReturnType<typeof setInterval> | null = null

function validateRegex(): boolean {
  regexError.value = null
  if (!regex.value) return true
  try {
    // eslint-disable-next-line no-new
    new RegExp(regex.value)
    return true
  } catch (e: any) {
    regexError.value = e?.message || 'Invalid regular expression'
    return false
  }
}

function currentQuery(since: number) {
  return {
    since,
    level: level.value || undefined,
    notebook: notebook.value.trim() || undefined,
    regex: regex.value || undefined,
    limit: FETCH_LIMIT,
  }
}

function extraFields(entry: LogEntry): [string, unknown][] {
  return Object.entries(entry).filter(([k]) => !KNOWN_FIELDS.has(k))
}

async function scrollToBottom() {
  await nextTick()
  const el = listEl.value
  if (el) el.scrollTop = el.scrollHeight
}

function onScroll() {
  const el = listEl.value
  if (!el) return
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  stickToBottom.value = atBottom
}

// Full reload: reset the buffer and fetch from the start. Used on mount,
// manual refresh, and whenever a filter changes.
async function reload() {
  if (!validateRegex()) return
  loading.value = true
  error.value = null
  try {
    const data = await strata.getLogs(currentQuery(0))
    entries.value = data.entries
    cursor.value = data.cursor
    expanded.value = new Set()
    await scrollToBottom()
    stickToBottom.value = true
  } catch (e: any) {
    error.value = e?.message || 'Failed to fetch logs'
  } finally {
    loading.value = false
  }
}

// Incremental tail: append only entries newer than the last cursor.
async function poll() {
  if (regexError.value) return
  try {
    const data = await strata.getLogs(currentQuery(cursor.value))
    if (data.entries.length) {
      entries.value = [...entries.value, ...data.entries].slice(-MAX_ENTRIES)
      if (stickToBottom.value) await scrollToBottom()
    }
    cursor.value = data.cursor
    error.value = null
  } catch (e: any) {
    error.value = e?.message || 'Failed to tail logs'
  }
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(() => void poll(), POLL_INTERVAL_MS)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

function toggleExpanded(c: number) {
  const next = new Set(expanded.value)
  if (next.has(c)) next.delete(c)
  else next.add(c)
  expanded.value = next
}

function clearFilters() {
  level.value = ''
  notebook.value = ''
  regex.value = ''
  regexError.value = null
}

const hasEntries = computed(() => entries.value.length > 0)

watch(liveTail, (on) => {
  if (on) startPolling()
  else stopPolling()
})

// Any filter change resets the view; if tailing, polling continues from the
// fresh cursor.
watch([level, notebook, regex], () => {
  void reload()
})

onMounted(() => {
  void reload()
})

onBeforeUnmount(() => {
  stopPolling()
})
</script>

<template>
  <div class="logs" data-testid="logs-page">
    <div class="logs-header">
      <div class="logs-title">
        <router-link to="/" class="back-link" title="Back to notebooks">◆ strata</router-link>
        <span class="crumb">/</span>
        <span class="page-name">logs</span>
      </div>
      <div class="logs-header-actions">
        <router-link to="/artifacts" class="nav-link" data-testid="nav-artifacts"
          >Artifacts</router-link
        >
        <ThemeToggle />
      </div>
    </div>

    <div class="filter-bar">
      <label class="filter">
        Level
        <select v-model="level" class="control" data-testid="logs-level">
          <option v-for="lvl in LEVELS" :key="lvl" :value="lvl">
            {{ lvl ? lvl.toUpperCase() : 'ALL' }}
          </option>
        </select>
      </label>
      <label class="filter">
        Notebook
        <input
          v-model="notebook"
          class="control"
          type="text"
          placeholder="notebook id"
          data-testid="logs-notebook"
        />
      </label>
      <label class="filter filter-grow">
        Message regex
        <input
          v-model="regex"
          class="control"
          :class="{ invalid: regexError }"
          type="text"
          placeholder="e.g. cache|error"
          data-testid="logs-regex"
        />
      </label>
      <label class="filter tail-toggle">
        <input v-model="liveTail" type="checkbox" data-testid="logs-tail" />
        Live tail
      </label>
      <button class="btn" :disabled="loading" data-testid="logs-refresh" @click="reload">
        Refresh
      </button>
      <button class="btn btn-secondary" @click="clearFilters">Clear</button>
    </div>

    <div v-if="regexError" class="banner banner-error" data-testid="logs-regex-error">
      Invalid regex: {{ regexError }}
    </div>
    <div v-else-if="error" class="banner banner-error">{{ error }}</div>

    <div ref="listEl" class="log-list" data-testid="logs-list" @scroll="onScroll">
      <div v-if="!hasEntries && !loading" class="empty">No log entries match these filters.</div>
      <div
        v-for="entry in entries"
        :key="entry.cursor"
        class="log-row"
        :class="`lvl-${(entry.level || 'info').toLowerCase()}`"
        :data-testid="`log-row-${entry.cursor}`"
        @click="toggleExpanded(entry.cursor)"
      >
        <span class="col-time">{{ entry.timestamp || '' }}</span>
        <span class="col-level">{{ (entry.level || 'info').toUpperCase() }}</span>
        <span class="col-logger">{{ entry.logger || '' }}</span>
        <span class="col-msg">{{ entry.message || '' }}</span>
        <div v-if="expanded.has(entry.cursor) && extraFields(entry).length" class="log-detail">
          <div v-for="[k, v] in extraFields(entry)" :key="k" class="detail-field">
            <span class="detail-key">{{ k }}</span>
            <span class="detail-val">{{ typeof v === 'string' ? v : JSON.stringify(v) }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="status-bar">
      <span>{{ entries.length }} shown</span>
      <span v-if="liveTail" class="tailing">● tailing</span>
      <span v-else-if="loading">loading…</span>
    </div>
  </div>
</template>

<style scoped>
.logs {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 16px 24px;
  gap: 12px;
}

.logs-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logs-title {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.logs-header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.nav-link {
  font-size: 13px;
  color: var(--text-secondary);
  text-decoration: none;
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
}

.nav-link:hover {
  color: var(--text-primary);
  border-color: var(--border-strong);
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

.control.invalid {
  border-color: var(--accent-danger);
}

.tail-toggle {
  flex-direction: row;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--text-primary);
  padding-bottom: 6px;
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

.log-list {
  flex: 1;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-surface);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
}

.empty {
  padding: 24px;
  text-align: center;
  color: var(--text-muted);
  font-family: inherit;
}

.log-row {
  display: grid;
  grid-template-columns: 180px 64px 180px 1fr;
  gap: 10px;
  padding: 3px 12px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  align-items: baseline;
}

.log-row:hover {
  background: var(--bg-hover);
}

.col-time {
  color: var(--text-muted);
  white-space: nowrap;
}

.col-level {
  font-weight: 600;
}

.col-logger {
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.col-msg {
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-word;
}

.lvl-error .col-level,
.lvl-critical .col-level {
  color: var(--accent-danger);
}

.lvl-warning .col-level {
  color: var(--accent-warning);
}

.lvl-info .col-level {
  color: var(--accent-primary);
}

.lvl-debug .col-level {
  color: var(--text-muted);
}

.log-detail {
  grid-column: 1 / -1;
  margin-top: 4px;
  padding: 6px 8px;
  background: var(--bg-elevated);
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.detail-field {
  display: flex;
  gap: 8px;
}

.detail-key {
  color: var(--accent-primary);
  min-width: 120px;
}

.detail-val {
  color: var(--text-primary);
  word-break: break-all;
}

.status-bar {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: var(--text-muted);
}

.tailing {
  color: var(--accent-primary);
}
</style>
