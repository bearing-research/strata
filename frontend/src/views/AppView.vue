<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { useNotebook } from '../stores/notebook'
import WidgetCell from '../components/WidgetCell.vue'
import DataTable from '../components/DataTable.vue'
import { renderMarkdownToHtml } from '../utils/markdown'
import type { Cell, CellOutput } from '../types/notebook'

/**
 * Read-only "app" view of a notebook: renders widget control panels + display
 * outputs only — no editor, DAG, or toolbars. The WS connects with
 * `?role=viewer` (mutations rejected server-side). Turn on a widget cell's
 * ⚡ Live toggle for the interactive "tweak a parameter, see the result" loop.
 */
const props = defineProps<{ sessionId: string }>()
const { notebook, orderedCells, openBySessionId, cleanupWebSocket, setViewerMode } = useNotebook()

const loading = ref(true)
const error = ref<string | null>(null)

// Embed mode (`/app/:id?embed=1`): drop the standalone chrome (title + Edit
// link) so the view drops cleanly into a host `<iframe>`, and post the content
// height to the parent frame so the host can size the iframe with no inner
// scrollbar. Only the height travels — no notebook data — so a `*` target
// origin is safe.
const route = useRoute()
const isEmbed = computed(() => route.query.embed === '1')
const rootEl = ref<HTMLElement | null>(null)
let resizeObserver: ResizeObserver | null = null

function postHeight() {
  if (!isEmbed.value || window.parent === window) return
  const height = Math.ceil(rootEl.value?.scrollHeight ?? document.body.scrollHeight)
  window.parent.postMessage(
    { type: 'strata:embed:resize', sessionId: props.sessionId, height },
    '*',
  )
}

onMounted(async () => {
  cleanupWebSocket() // drop any editor connection so we reconnect read-only
  setViewerMode(true)
  try {
    await openBySessionId(props.sessionId)
  } catch (e) {
    error.value = (e as Error).message || 'Failed to open notebook'
  } finally {
    loading.value = false
  }
  if (isEmbed.value) {
    await nextTick()
    resizeObserver = new ResizeObserver(() => postHeight())
    if (rootEl.value) resizeObserver.observe(rootEl.value)
    postHeight()
  }
})

onUnmounted(() => {
  setViewerMode(false)
  cleanupWebSocket()
  resizeObserver?.disconnect()
})

const notebookName = computed(() => notebook.name || 'Notebook')

// Cells worth showing in an app: widget panels, markdown prose, and anything
// with a display output. Bare compute cells with no output — and cells the
// author marked `# @app hide` — are hidden.
function isAppHidden(source: string): boolean {
  return source.split('\n').some((line) => /^#\s*@app\s+hide\b/.test(line.trim()))
}
const appCells = computed(() =>
  orderedCells.value.filter(
    (c) =>
      !isAppHidden(c.source) &&
      (c.language === 'widget' ||
        c.language === 'markdown' ||
        (c.displayOutputs && c.displayOutputs.length > 0)),
  ),
)

function cellTitle(cell: Cell): string {
  return cell.annotations?.name || ''
}

// A preview row arrives as a column-keyed dict; DataTable wants positional rows.
function outputRows(output: CellOutput): unknown[][] {
  const cols = output.columns ?? []
  return (output.rows ?? []).map((row) => cols.map((c) => (row as Record<string, unknown>)[c]))
}
</script>

<template>
  <div ref="rootEl" class="app-view" :class="{ embed: isEmbed }">
    <header v-if="!isEmbed" class="app-header">
      <h1>{{ notebookName }}</h1>
      <router-link class="edit-link" :to="`/notebook/${sessionId}`">← Edit</router-link>
    </header>

    <div v-if="loading" class="app-status">Loading…</div>
    <div v-else-if="error" class="app-status error">{{ error }}</div>
    <div v-else-if="!appCells.length" class="app-status">
      This notebook has no widgets or outputs to display yet.
    </div>

    <div v-else class="app-body">
      <section v-for="cell in appCells" :key="cell.id" class="app-cell">
        <h2 v-if="cellTitle(cell)" class="app-cell-title">{{ cellTitle(cell) }}</h2>

        <!-- Widget control panel (interactive) -->
        <WidgetCell v-if="cell.language === 'widget'" :cell="cell" />

        <!-- Markdown prose -->
        <div
          v-else-if="cell.language === 'markdown'"
          class="app-markdown"
          v-html="renderMarkdownToHtml(cell.source)"
        ></div>

        <!-- Display outputs -->
        <template v-else>
          <div v-for="(output, i) in cell.displayOutputs || []" :key="i" class="app-output">
            <DataTable
              v-if="output.contentType === 'arrow/ipc' && output.columns?.length"
              :notebook-id="sessionId"
              :cell-id="cell.id"
              :artifact-uri="output.artifactUri || ''"
              :columns="output.columns"
              :preview-rows="outputRows(output)"
              :total="output.rowCount ?? output.rows?.length ?? 0"
            />
            <img
              v-else-if="output.contentType === 'image/png' && output.inlineDataUrl"
              :src="output.inlineDataUrl"
              alt="output"
              class="app-image"
            />
            <div
              v-else-if="output.contentType === 'text/markdown' && output.markdownText"
              class="app-markdown"
              v-html="renderMarkdownToHtml(output.markdownText)"
            ></div>
            <pre v-else-if="output.scalar !== undefined" class="app-scalar">{{
              output.scalar
            }}</pre>
          </div>
        </template>
      </section>
    </div>
  </div>
</template>

<style scoped>
.app-view {
  max-width: 900px;
  margin: 0 auto;
  padding: 24px 20px 64px;
}
/* Embedded in a host iframe: tighter padding, no full-page bottom gutter, and
   let the host's background show through rather than forcing our own. */
.app-view.embed {
  padding: 12px 14px;
  max-width: none;
  background: transparent;
}
.app-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  border-bottom: 1px solid var(--border);
  padding-bottom: 12px;
  margin-bottom: 24px;
}
.app-header h1 {
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
}
.edit-link {
  color: var(--text-muted);
  text-decoration: none;
  font-size: 13px;
}
.edit-link:hover {
  color: var(--accent-primary);
}
.app-status {
  color: var(--text-muted);
  padding: 40px 0;
  text-align: center;
}
.app-status.error {
  color: var(--accent-danger);
}
.app-cell {
  margin-bottom: 28px;
}
.app-cell-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 10px;
}
.app-image {
  max-width: 100%;
  display: block;
}
.app-scalar {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: var(--text-primary);
  white-space: pre-wrap;
}
.app-markdown {
  color: var(--text-primary);
  line-height: 1.6;
}
</style>
