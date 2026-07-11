<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useStrata } from '../composables/useStrata'

/**
 * Interactive viewer for a DataFrame cell output.
 *
 * The cell payload only carries a 20-row preview; this component queries the
 * *full* cached Arrow artifact via `GET …/cells/{cellId}/data` for paging,
 * sorting, global search, per-column filters, a column-stats row, and
 * CSV/Parquet export. The preview paints instantly; server queries kick in
 * when there's a live session + backing artifact URI.
 */
const props = defineProps<{
  notebookId: string | undefined
  cellId: string
  artifactUri: string
  columns: string[]
  previewRows: unknown[][]
  total: number
}>()

const strata = useStrata()
const PAGE_SIZES = [25, 50, 100, 250]
const OPS = [
  { v: 'contains', label: 'contains' },
  { v: 'eq', label: '=' },
  { v: 'ne', label: '≠' },
  { v: 'gt', label: '>' },
  { v: 'ge', label: '≥' },
  { v: 'lt', label: '<' },
  { v: 'le', label: '≤' },
  { v: 'between', label: 'between' },
  { v: 'is_null', label: 'is null' },
  { v: 'not_null', label: 'not null' },
]

interface FilterSpec {
  col: string
  op: string
  value?: unknown
  value2?: unknown
}

const columns = ref<string[]>(props.columns)
const rows = ref<unknown[][]>(props.previewRows)
const total = ref<number>(props.total)
const offset = ref(0)
const pageSize = ref(50)
const sortBy = ref<string | null>(null)
const sortDir = ref<'asc' | 'desc'>('asc')
const search = ref('')
const filters = ref<FilterSpec[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

const showStats = ref(false)
const summary = ref<
  { name: string; dtype: string; nulls: number; distinct: number; min: unknown; max: unknown }[]
>([])
const summaryByName = computed(() => Object.fromEntries(summary.value.map((s) => [s.name, s])))

const expanded = ref<Set<string>>(new Set())
const showFilterForm = ref(false)
const draft = reactive({ col: '', op: 'contains', value: '', value2: '' })

// Server queries need a live session + a backing artifact; without them
// (e.g. mock data) we only show the static preview.
const canQuery = computed(() => Boolean(props.notebookId) && Boolean(props.artifactUri))
const hasQuery = computed(() => Boolean(search.value) || filters.value.length > 0)
const rangeStart = computed(() => (rows.value.length ? offset.value + 1 : 0))
const rangeEnd = computed(() => offset.value + rows.value.length)
const hasPrev = computed(() => canQuery.value && offset.value > 0)
const hasNext = computed(() => canQuery.value && offset.value + rows.value.length < total.value)
const showPager = computed(
  () => canQuery.value && (total.value > props.previewRows.length || hasQuery.value),
)

function queryOpts() {
  return {
    sortBy: sortBy.value,
    sortDir: sortDir.value,
    search: search.value || null,
    filters: filters.value,
  }
}

async function fetchPage() {
  if (!props.notebookId || !props.artifactUri) return
  loading.value = true
  error.value = null
  try {
    const page = await strata.getCellData(props.notebookId, props.cellId, props.artifactUri, {
      offset: offset.value,
      limit: pageSize.value,
      ...queryOpts(),
    })
    if (page.pageable) {
      columns.value = page.columns
      rows.value = page.rows
      total.value = page.total
      expanded.value = new Set()
    }
  } catch (e) {
    error.value = (e as Error).message || 'Failed to load rows'
  } finally {
    loading.value = false
  }
}

/** Re-run a query from the first page (after sort / search / filter change). */
function applyQuery() {
  offset.value = 0
  fetchPage()
}

function prevPage() {
  if (!hasPrev.value) return
  offset.value = Math.max(0, offset.value - pageSize.value)
  fetchPage()
}

function nextPage() {
  if (!hasNext.value) return
  offset.value = offset.value + pageSize.value
  fetchPage()
}

function changePageSize(event: Event) {
  pageSize.value = Number((event.target as HTMLSelectElement).value)
  applyQuery()
}

function sortByColumn(col: string) {
  if (!canQuery.value) return
  if (sortBy.value !== col) {
    sortBy.value = col
    sortDir.value = 'asc'
  } else if (sortDir.value === 'asc') {
    sortDir.value = 'desc'
  } else {
    sortBy.value = null // third click clears the sort
  }
  applyQuery()
}

function sortIndicator(col: string): string {
  if (sortBy.value !== col) return ''
  return sortDir.value === 'asc' ? ' ▲' : ' ▼'
}

// Debounced search — refetch 300ms after the user stops typing.
let searchTimer: ReturnType<typeof setTimeout> | undefined
watch(search, () => {
  if (!canQuery.value) return
  clearTimeout(searchTimer)
  searchTimer = setTimeout(applyQuery, 300)
})

function needsValue(op: string) {
  return op !== 'is_null' && op !== 'not_null'
}
function needsValue2(op: string) {
  return op === 'between'
}

function addFilter() {
  if (!draft.col) return
  const spec: FilterSpec = { col: draft.col, op: draft.op }
  if (needsValue(draft.op)) spec.value = draft.value
  if (needsValue2(draft.op)) spec.value2 = draft.value2
  filters.value = [...filters.value, spec]
  showFilterForm.value = false
  draft.value = ''
  draft.value2 = ''
  applyQuery()
}

function removeFilter(index: number) {
  filters.value = filters.value.filter((_, i) => i !== index)
  applyQuery()
}

function filterLabel(f: FilterSpec): string {
  const op = OPS.find((o) => o.v === f.op)?.label ?? f.op
  if (!needsValue(f.op)) return `${f.col} ${op}`
  if (needsValue2(f.op)) return `${f.col} ${op} ${f.value}…${f.value2}`
  return `${f.col} ${op} ${f.value}`
}

async function toggleStats() {
  showStats.value = !showStats.value
  if (showStats.value && !summary.value.length && props.notebookId) {
    try {
      summary.value = await strata.getCellDataSummary(
        props.notebookId,
        props.cellId,
        props.artifactUri,
      )
    } catch (e) {
      error.value = (e as Error).message || 'Failed to load column stats'
    }
  }
}

function exportUrl(fmt: 'csv' | 'parquet'): string {
  return strata.cellDataExportUrl(
    props.notebookId ?? '',
    props.cellId,
    props.artifactUri,
    fmt,
    queryOpts(),
  )
}

// Cell rendering ----------------------------------------------------------
function isNull(v: unknown): boolean {
  return v === null || v === undefined
}
function isNumeric(v: unknown): boolean {
  return typeof v === 'number' && Number.isFinite(v)
}
function fmtCell(v: unknown): string {
  if (isNull(v)) return 'null'
  if (typeof v === 'number') {
    // Leave small integers (ids, years) unformatted; group larger/real numbers.
    if (Number.isInteger(v) && Math.abs(v) < 10000) return String(v)
    return v.toLocaleString(undefined, { maximumFractionDigits: 6 })
  }
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}
function cellKey(r: number, c: number): string {
  return `${r}:${c}`
}
function toggleCell(r: number, c: number) {
  const key = cellKey(r, c)
  const next = new Set(expanded.value)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  expanded.value = next
}

// Re-run replaces the backing artifact — reset all query state to the preview.
watch(
  () => props.artifactUri,
  () => {
    columns.value = props.columns
    rows.value = props.previewRows
    total.value = props.total
    offset.value = 0
    sortBy.value = null
    sortDir.value = 'asc'
    search.value = ''
    filters.value = []
    summary.value = []
    expanded.value = new Set()
    error.value = null
    if (canQuery.value && total.value > props.previewRows.length) fetchPage()
  },
)

onMounted(() => {
  if (canQuery.value && total.value > props.previewRows.length) fetchPage()
})
</script>

<template>
  <div class="data-table" :class="{ loading }">
    <div v-if="canQuery" class="data-table-toolbar">
      <input
        v-model="search"
        class="search-input"
        type="search"
        placeholder="Search all columns…"
      />
      <div class="toolbar-actions">
        <button class="tool-btn" :class="{ on: showStats }" @click="toggleStats">Stats</button>
        <button class="tool-btn" @click="showFilterForm = !showFilterForm">+ Filter</button>
        <a class="tool-btn" :href="exportUrl('csv')" download>CSV</a>
        <a class="tool-btn" :href="exportUrl('parquet')" download>Parquet</a>
      </div>
    </div>

    <div v-if="showFilterForm && canQuery" class="filter-form">
      <select v-model="draft.col">
        <option value="" disabled>Column…</option>
        <option v-for="col in columns" :key="col" :value="col">{{ col }}</option>
      </select>
      <select v-model="draft.op">
        <option v-for="op in OPS" :key="op.v" :value="op.v">{{ op.label }}</option>
      </select>
      <input v-if="needsValue(draft.op)" v-model="draft.value" placeholder="value" />
      <input v-if="needsValue2(draft.op)" v-model="draft.value2" placeholder="and…" />
      <button class="tool-btn" :disabled="!draft.col" @click="addFilter">Add</button>
    </div>

    <div v-if="filters.length" class="filter-chips">
      <span v-for="(f, i) in filters" :key="i" class="chip">
        {{ filterLabel(f) }}
        <button class="chip-x" @click="removeFilter(i)">✕</button>
      </span>
    </div>

    <div class="data-table-scroll">
      <table>
        <thead>
          <tr>
            <th
              v-for="col in columns"
              :key="col"
              :class="{ sortable: canQuery, active: sortBy === col }"
              @click="sortByColumn(col)"
            >
              {{ col }}<span class="sort-caret">{{ sortIndicator(col) }}</span>
            </th>
          </tr>
          <tr v-if="showStats" class="stats-row">
            <th v-for="col in columns" :key="col">
              <template v-if="summaryByName[col]">
                <span class="dtype">{{ summaryByName[col].dtype }}</span>
                <span v-if="summaryByName[col].nulls" class="stat"
                  >{{ summaryByName[col].nulls }} null</span
                >
                <span class="stat">{{ summaryByName[col].distinct }} uniq</span>
                <span v-if="summaryByName[col].min !== null" class="stat range">
                  {{ fmtCell(summaryByName[col].min) }} – {{ fmtCell(summaryByName[col].max) }}
                </span>
              </template>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, rowIndex) in rows" :key="rowIndex">
            <td
              v-for="(col, colIndex) in columns"
              :key="col"
              :class="{
                numeric: isNumeric(row[colIndex]),
                'null-cell': isNull(row[colIndex]),
                expanded: expanded.has(cellKey(rowIndex, colIndex)),
              }"
              :title="fmtCell(row[colIndex])"
              @click="toggleCell(rowIndex, colIndex)"
            >
              {{ fmtCell(row[colIndex]) }}
            </td>
          </tr>
          <tr v-if="!rows.length">
            <td :colspan="columns.length || 1" class="empty">No matching rows</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="data-table-footer">
      <span class="range">
        <template v-if="total">
          {{ rangeStart.toLocaleString() }}–{{ rangeEnd.toLocaleString() }} of
          {{ total.toLocaleString() }} rows
        </template>
        <template v-else>0 rows</template>
      </span>
      <span v-if="error" class="data-table-error">{{ error }}</span>
      <span v-else-if="loading" class="data-table-loading">Loading…</span>
      <span v-if="showPager" class="pager">
        <label
          >Rows
          <select :value="pageSize" @change="changePageSize">
            <option v-for="size in PAGE_SIZES" :key="size" :value="size">{{ size }}</option>
          </select>
        </label>
        <button :disabled="!hasPrev || loading" @click="prevPage">‹ Prev</button>
        <button :disabled="!hasNext || loading" @click="nextPage">Next ›</button>
      </span>
    </div>
  </div>
</template>

<style scoped>
.data-table {
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
}
.data-table-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border);
}
.search-input {
  flex: 1;
  min-width: 0;
  background: var(--bg-input);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 3px 8px;
  font: inherit;
}
.toolbar-actions {
  display: flex;
  gap: 6px;
}
.tool-btn {
  background: var(--bg-input);
  color: var(--text-secondary);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 3px 8px;
  cursor: pointer;
  text-decoration: none;
  font: inherit;
  line-height: 1.4;
}
.tool-btn:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}
.tool-btn.on {
  color: var(--accent-primary);
  border-color: var(--accent-primary);
}
.filter-form {
  display: flex;
  gap: 6px;
  padding: 6px 8px;
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}
.filter-form select,
.filter-form input {
  background: var(--bg-input);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 2px 6px;
  font: inherit;
}
.filter-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 6px 8px;
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border);
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1px 4px 1px 8px;
  color: var(--text-primary);
}
.chip-x {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 0 2px;
  font-size: 10px;
}
.chip-x:hover {
  color: var(--accent-danger);
}
.data-table-scroll {
  overflow: auto;
  max-height: 420px;
}
.data-table table {
  border-collapse: collapse;
  width: 100%;
}
.data-table th,
.data-table td {
  padding: 4px 12px;
  text-align: left;
  white-space: nowrap;
  border-bottom: 1px solid var(--border-subtle);
  max-width: 320px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.data-table td.expanded {
  white-space: normal;
  overflow: visible;
  max-width: none;
  word-break: break-word;
}
.data-table td.numeric {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.data-table td.null-cell {
  color: var(--text-muted);
  font-style: italic;
}
.data-table td.empty {
  text-align: center;
  color: var(--text-muted);
  padding: 12px;
}
.data-table thead th {
  position: sticky;
  top: 0;
  background: var(--bg-elevated);
  color: var(--accent-primary);
  font-weight: 600;
  user-select: none;
}
.data-table th.sortable {
  cursor: pointer;
}
.data-table th.sortable:hover {
  background: var(--bg-hover);
}
.data-table th.active {
  color: var(--accent-primary-hover);
}
.sort-caret {
  font-size: 10px;
}
.stats-row th {
  top: 26px;
  font-weight: 400;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
}
.stats-row .dtype {
  color: var(--accent-teal);
  margin-right: 6px;
}
.stats-row .stat {
  margin-right: 6px;
}
.stats-row .range {
  white-space: nowrap;
}
.data-table td {
  color: var(--text-primary);
  cursor: default;
}
.data-table tbody tr:hover td {
  background: var(--bg-input);
}
.data-table-footer {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 12px;
  background: var(--bg-surface);
  border-top: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 11px;
}
.data-table-footer .range {
  flex: 1;
}
.data-table-error {
  color: var(--accent-danger);
}
.pager {
  display: flex;
  align-items: center;
  gap: 8px;
}
.pager select {
  background: var(--bg-input);
  color: inherit;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 4px;
}
.pager button {
  background: var(--bg-input);
  color: inherit;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 2px 8px;
  cursor: pointer;
}
.pager button:disabled {
  opacity: 0.4;
  cursor: default;
}
</style>
