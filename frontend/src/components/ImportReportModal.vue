<script setup lang="ts">
/**
 * Post-import dialog: shows the conversion report and lets the user
 * either open the new notebook (the normal flow) or close the dialog
 * and stay on the home page (the "I just wanted to check the report"
 * flow).
 *
 * Used by the Jupyter-import flow on HomePage. The report itself is
 * server-rendered as Markdown; we run it through the same
 * renderMarkdownToHtml() helper the rest of the UI uses, which
 * goes through markdown-it + DOMPurify.
 */
import { computed } from 'vue'
import type { ImportReport } from '../composables/useStrata'
import { renderMarkdownToHtml } from '../utils/markdown'

const props = defineProps<{
  report: ImportReport
  notebookName: string
}>()

const emit = defineEmits<{
  (e: 'open'): void
  (e: 'close'): void
}>()

const reportHtml = computed(() => renderMarkdownToHtml(props.report.report_text || ''))

const hasWarnings = computed(() => props.report.warnings.length > 0)
const hasDropped = computed(
  () => props.report.dropped_magics.length > 0 || props.report.dropped_shells.length > 0,
)
</script>

<template>
  <div class="import-report-backdrop" @click.self="emit('close')">
    <div class="import-report-modal" role="dialog" aria-labelledby="import-report-title">
      <header class="import-report-header">
        <h2 id="import-report-title">Imported {{ notebookName }}</h2>
        <button class="import-report-close" type="button" aria-label="Close" @click="emit('close')">
          ×
        </button>
      </header>

      <div class="import-report-summary">
        <span class="import-report-stat">
          <strong>{{ report.code_cells }}</strong> code
        </span>
        <span class="import-report-stat">
          <strong>{{ report.markdown_cells }}</strong> markdown
        </span>
        <span v-if="report.captured_deps.length" class="import-report-stat">
          <strong>{{ report.captured_deps.length }}</strong> deps captured
        </span>
        <span v-if="report.translated_magics.length" class="import-report-stat">
          <strong>{{ report.translated_magics.length }}</strong> magics translated
        </span>
        <span v-if="hasDropped" class="import-report-stat import-report-stat-warn">
          <strong>{{ report.dropped_magics.length + report.dropped_shells.length }}</strong>
          dropped
        </span>
        <span v-if="hasWarnings" class="import-report-stat import-report-stat-warn">
          <strong>{{ report.warnings.length }}</strong> warning{{
            report.warnings.length === 1 ? '' : 's'
          }}
        </span>
      </div>

      <div class="import-report-body" v-html="reportHtml" />

      <footer class="import-report-actions">
        <button class="btn btn-secondary" type="button" @click="emit('close')">Close</button>
        <button class="btn" type="button" data-testid="import-open-notebook" @click="emit('open')">
          Open notebook
        </button>
      </footer>
    </div>
  </div>
</template>

<style scoped>
.import-report-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 1.5rem;
}

.import-report-modal {
  background: var(--bg-surface, #fff);
  color: var(--text-primary, #111);
  border-radius: 8px;
  max-width: 720px;
  width: 100%;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.25);
}

.import-report-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--border-color, #e5e7eb);
}

.import-report-header h2 {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 600;
}

.import-report-close {
  background: transparent;
  border: 0;
  font-size: 1.4rem;
  line-height: 1;
  cursor: pointer;
  color: var(--text-secondary, #6b7280);
  padding: 0 0.25rem;
}

.import-report-close:hover {
  color: var(--text-primary, #111);
}

.import-report-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem 1rem;
  padding: 0.75rem 1.25rem;
  background: var(--bg-muted, #f9fafb);
  border-bottom: 1px solid var(--border-color, #e5e7eb);
  font-size: 0.9rem;
}

.import-report-stat strong {
  font-weight: 600;
}

.import-report-stat-warn {
  color: var(--text-warning, #b45309);
}

.import-report-body {
  padding: 1rem 1.25rem;
  overflow-y: auto;
  flex: 1;
  font-size: 0.9rem;
  line-height: 1.5;
}

.import-report-body :deep(h1),
.import-report-body :deep(h2),
.import-report-body :deep(h3) {
  margin-top: 1.25em;
  margin-bottom: 0.5em;
}

.import-report-body :deep(h1) {
  font-size: 1.15rem;
}

.import-report-body :deep(h2) {
  font-size: 1.05rem;
}

.import-report-body :deep(h3) {
  font-size: 0.95rem;
}

.import-report-body :deep(ul) {
  padding-left: 1.5em;
  margin: 0.5em 0;
}

.import-report-body :deep(code) {
  background: var(--bg-muted, #f3f4f6);
  padding: 0.1em 0.35em;
  border-radius: 3px;
  font-size: 0.85em;
}

.import-report-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  padding: 0.75rem 1.25rem;
  border-top: 1px solid var(--border-color, #e5e7eb);
}
</style>
