<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useStrata } from '../composables/useStrata'

/**
 * Interactive viewer for a DataFrame cell output.
 *
 * The cell payload only carries a 20-row preview; this component lazily
 * pages/sorts the *full* cached Arrow artifact via
 * ``GET /v1/notebooks/{id}/cells/{cellId}/data``. The preview paints
 * instantly on first render; when the frame is larger than the preview
 * (and a backing artifact URI exists) the first real page is fetched so
 * paging stays consistent.
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

const columns = ref<string[]>(props.columns)
const rows = ref<unknown[][]>(props.previewRows)
const total = ref<number>(props.total)
const offset = ref(0)
const pageSize = ref(50)
const sortBy = ref<string | null>(null)
const sortDir = ref<'asc' | 'desc'>('asc')
const loading = ref(false)
const error = ref<string | null>(null)

// Paging needs a live server session and a backing artifact; without either
// (e.g. mock data) we just show the static preview.
const canPage = computed(
  () =>
    Boolean(props.notebookId) &&
    Boolean(props.artifactUri) &&
    total.value > props.previewRows.length,
)
const rangeStart = computed(() => (rows.value.length ? offset.value + 1 : 0))
const rangeEnd = computed(() => offset.value + rows.value.length)
const hasPrev = computed(() => canPage.value && offset.value > 0)
const hasNext = computed(() => canPage.value && offset.value + rows.value.length < total.value)

async function fetchPage() {
  if (!props.notebookId || !props.artifactUri) return
  loading.value = true
  error.value = null
  try {
    const page = await strata.getCellData(props.notebookId, props.cellId, props.artifactUri, {
      offset: offset.value,
      limit: pageSize.value,
      sortBy: sortBy.value,
      sortDir: sortDir.value,
    })
    if (page.pageable) {
      columns.value = page.columns
      rows.value = page.rows
      total.value = page.total
    }
  } catch (e) {
    error.value = (e as Error).message || 'Failed to load rows'
  } finally {
    loading.value = false
  }
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
  offset.value = 0
  fetchPage()
}

function sortByColumn(col: string) {
  if (!canPage.value) return
  if (sortBy.value !== col) {
    sortBy.value = col
    sortDir.value = 'asc'
  } else if (sortDir.value === 'asc') {
    sortDir.value = 'desc'
  } else {
    sortBy.value = null // third click clears the sort
  }
  offset.value = 0
  fetchPage()
}

function sortIndicator(col: string): string {
  if (sortBy.value !== col) return ''
  return sortDir.value === 'asc' ? ' ▲' : ' ▼'
}

function fmt(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

// Re-run replaces the backing artifact — reset to the fresh preview.
watch(
  () => props.artifactUri,
  () => {
    columns.value = props.columns
    rows.value = props.previewRows
    total.value = props.total
    offset.value = 0
    sortBy.value = null
    sortDir.value = 'asc'
    error.value = null
    if (canPage.value) fetchPage()
  },
)

onMounted(() => {
  if (canPage.value) fetchPage()
})
</script>

<template>
  <div class="data-table" :class="{ loading }">
    <div class="data-table-scroll">
      <table>
        <thead>
          <tr>
            <th
              v-for="col in columns"
              :key="col"
              :class="{ sortable: canPage, active: sortBy === col }"
              @click="sortByColumn(col)"
            >
              {{ col }}<span class="sort-caret">{{ sortIndicator(col) }}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, rowIndex) in rows" :key="rowIndex">
            <td v-for="(col, colIndex) in columns" :key="col">{{ fmt(row[colIndex]) }}</td>
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
      <span v-if="canPage" class="pager">
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
.data-table td {
  color: var(--text-primary);
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
