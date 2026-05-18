<script setup lang="ts">
import { computed, ref } from 'vue'
import { useNotebook } from '../stores/notebook'
import type { Cell, CellId } from '../types/notebook'

const { orderedCells } = useNotebook()

interface NoteCard {
  id: CellId
  title: string
  preview: string
}

function firstMarkdownHeading(source: string): string | null {
  for (const raw of source.split('\n')) {
    const line = raw.trim()
    const m = /^#+\s+(.+)$/.exec(line)
    if (m) return m[1].trim()
  }
  return null
}

function cardFor(cell: Cell): NoteCard {
  // Title priority:
  //   1. ``# name`` annotation (matches the cell-name convention used by
  //      DagView for compute cells)
  //   2. First markdown heading inside the body
  //   3. Cell ID prefix (deterministic fallback)
  const annotationName = cell.annotations?.name ?? null
  const headingName = firstMarkdownHeading(cell.source)
  const title = annotationName || headingName || cell.id.slice(0, 8)
  // Preview is the first non-blank line *after* the title source so it
  // doesn't duplicate the title; falls back to "(empty)" for blank cells.
  const lines = cell.source.split('\n').map((l) => l.trim())
  const titleLine = annotationName ? '' : (headingName ?? '')
  const previewLine = lines.find((l) => l && l !== `# ${titleLine}` && !l.startsWith('# ')) ?? ''
  const preview = previewLine || '(empty)'
  return { id: cell.id, title, preview }
}

const noteCards = computed<NoteCard[]>(() =>
  orderedCells.value.filter((c) => c.language === 'markdown').map(cardFor),
)

const collapsed = ref(false)

function scrollToCell(cellId: CellId) {
  // Reuses the data-cell-id attribute the cell editor list already
  // stamps; matches DagView.vue:229. Kept inline rather than emitting
  // because the editor area is a sibling tree we'd otherwise need an
  // event bus to reach.
  const el = document.querySelector(`[data-testid="notebook-cell"][data-cell-id="${cellId}"]`)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    el.classList.add('dag-jump-highlight')
    setTimeout(() => el.classList.remove('dag-jump-highlight'), 1500)
  }
}
</script>

<template>
  <aside class="notes-panel" :class="{ collapsed }" data-testid="notes-panel">
    <header class="notes-panel-header" @click="collapsed = !collapsed">
      <span class="notes-panel-title">Notes</span>
      <span class="notes-panel-count">{{ noteCards.length }}</span>
      <span class="notes-panel-chevron">{{ collapsed ? '▸' : '▾' }}</span>
    </header>
    <div v-if="!collapsed" class="notes-panel-body">
      <div v-if="noteCards.length === 0" class="notes-panel-empty">No markdown cells</div>
      <button
        v-for="card in noteCards"
        :key="card.id"
        class="note-card"
        type="button"
        :title="card.preview"
        @click="scrollToCell(card.id)"
      >
        <div class="note-card-title">{{ card.title }}</div>
        <div class="note-card-preview">{{ card.preview }}</div>
      </button>
    </div>
  </aside>
</template>

<style scoped>
.notes-panel {
  width: 240px;
  flex-shrink: 0;
  border-left: 1px solid var(--border-subtle);
  background: var(--bg-subtle);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-size: 12px;
}
.notes-panel.collapsed {
  width: 32px;
}
.notes-panel-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--border-subtle);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: 11px;
  color: var(--text-muted);
}
.notes-panel-header:hover {
  color: var(--text);
}
.notes-panel-title {
  flex: 1;
}
.notes-panel.collapsed .notes-panel-title,
.notes-panel.collapsed .notes-panel-count {
  display: none;
}
.notes-panel-count {
  background: var(--cat-surface1, var(--border-subtle));
  color: var(--text-muted);
  padding: 1px 6px;
  border-radius: 8px;
  font-size: 10px;
  letter-spacing: 0;
  text-transform: none;
}
.notes-panel-chevron {
  color: var(--text-muted);
  font-size: 10px;
}
.notes-panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.notes-panel-empty {
  color: var(--text-muted);
  font-style: italic;
  padding: 12px 4px;
  text-align: center;
}
.note-card {
  text-align: left;
  background: var(--bg);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  padding: 8px 10px;
  cursor: pointer;
  color: inherit;
  font: inherit;
  display: flex;
  flex-direction: column;
  gap: 4px;
  transition:
    border-color 0.12s ease,
    background-color 0.12s ease;
}
.note-card:hover {
  border-color: var(--accent-primary);
  background: var(--tint-primary, var(--bg-subtle));
}
.note-card-title {
  font-weight: 500;
  font-size: 12px;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.note-card-preview {
  color: var(--text-muted);
  font-size: 11px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
