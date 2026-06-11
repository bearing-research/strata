<script setup lang="ts">
import { useNotebook } from '../stores/notebook'

const { lineageOpen, lineageTitle, lineageRows, lineageLoading, lineageError, closeLineage } =
  useNotebook()
</script>

<template>
  <div v-if="lineageOpen" class="lineage-overlay" @click.self="closeLineage">
    <div class="lineage-modal">
      <header class="lineage-header">
        <span class="lineage-title">Lineage · {{ lineageTitle }}</span>
        <button class="lineage-close" @click="closeLineage">✕</button>
      </header>
      <div class="lineage-body">
        <div v-if="lineageLoading" class="lineage-muted">Loading…</div>
        <div v-else-if="lineageError" class="lineage-error">{{ lineageError }}</div>
        <div v-else-if="!lineageRows.length" class="lineage-muted">No lineage recorded.</div>
        <div v-else class="lineage-tree">
          <div
            v-for="(row, i) in lineageRows"
            :key="`${row.uri}:${i}`"
            class="lineage-row"
            :style="{ paddingLeft: row.depth * 18 + 'px' }"
          >
            <span class="branch">{{ row.depth === 0 ? '' : '└─' }}</span>
            <span class="ltype" :class="row.type">{{ row.type === 'table' ? '⛁' : '⬡' }}</span>
            <span class="llabel">{{ row.label }}</span>
            <span v-if="row.version != null" class="lver">v{{ row.version }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.lineage-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3000;
}
.lineage-modal {
  background: var(--bg, #fff);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  min-width: 420px;
  max-width: 680px;
  max-height: 70vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
}
.lineage-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border-subtle);
}
.lineage-title {
  font-weight: 600;
  color: var(--text);
}
.lineage-close {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  color: var(--text-muted);
}
.lineage-body {
  padding: 12px 14px;
  overflow: auto;
}
.lineage-tree {
  font-family: var(--font-mono, monospace);
  font-size: 12px;
}
.lineage-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 2px 0;
}
.branch {
  color: var(--text-muted);
}
.ltype.table {
  color: var(--accent-warning, #d4a017);
}
.ltype.artifact {
  color: var(--accent-primary, #3b82f6);
}
.llabel {
  color: var(--text);
}
.lver {
  color: var(--text-muted);
}
.lineage-muted {
  color: var(--text-muted);
}
.lineage-error {
  color: var(--accent-danger, #c0392b);
}
</style>
