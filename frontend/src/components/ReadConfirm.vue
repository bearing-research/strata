<script setup lang="ts">
import { computed } from 'vue'
import { useNotebook } from '../stores/notebook'
import { describeInput, formatBytes } from '../utils/inputWeights'

const { pendingReadConfirm, confirmReads, cancelReads, cellMap } = useNotebook()

function labelFor(cellId: string): string {
  const cell = cellMap.value.get(cellId)
  return cell?.annotations?.name || cellId.slice(0, 8)
}

const cells = computed(() =>
  (pendingReadConfirm.value?.cells ?? []).map((cell) => ({
    ...cell,
    label: labelFor(cell.cell_id),
    moves: formatBytes(cell.bytes_to_move),
    cost: cell.cost > 0 ? `about $${cell.cost.toFixed(2)}` : null,
    inputs: cell.inputs.map(describeInput),
  })),
)

const message = computed(() => pendingReadConfirm.value?.message ?? null)
const messageIsLink = computed(() => /^https?:\/\/\S+$/.test(message.value ?? ''))
</script>

<template>
  <div v-if="pendingReadConfirm" class="read-overlay" role="dialog" aria-modal="true">
    <div class="read-card">
      <div class="read-title">Read this much before running?</div>
      <ul class="read-cells">
        <li v-for="cell in cells" :key="cell.cell_id">
          <div class="read-cell">
            <code>{{ cell.label }}</code> moves {{ cell.moves }} onto <code>{{ cell.machine }}</code
            ><template v-if="cell.cost">, {{ cell.cost }}</template>
          </div>
          <ul class="read-inputs">
            <li v-for="line in cell.inputs" :key="line">{{ line }}</li>
          </ul>
        </li>
      </ul>
      <div v-if="message" class="read-message">
        <a v-if="messageIsLink" :href="message" target="_blank" rel="noopener noreferrer">{{
          message
        }}</a>
        <template v-else>{{ message }}</template>
      </div>
      <div class="read-actions">
        <button class="read-cancel" @click="cancelReads">Cancel</button>
        <button class="read-confirm" @click="confirmReads">Run</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.read-overlay {
  position: fixed;
  inset: 0;
  background: rgba(10, 10, 20, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.read-card {
  background: var(--bg-elevated);
  border: 1px solid var(--accent-primary);
  border-radius: 8px;
  padding: 18px 20px;
  min-width: 340px;
  max-width: 560px;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
}
.read-title {
  color: var(--text-primary);
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 8px;
}
.read-cells,
.read-inputs {
  list-style: none;
  margin: 0;
  padding: 0;
}
.read-cell {
  color: var(--text-primary);
  font-size: 13px;
  margin-top: 6px;
}
.read-inputs {
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.45;
  padding-left: 12px;
  overflow-wrap: anywhere;
}
.read-cell code {
  color: var(--accent-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.read-message {
  color: var(--text-secondary);
  font-size: 13px;
  line-height: 1.45;
  margin-top: 12px;
}
.read-message a {
  color: var(--accent-primary);
}
.read-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 14px;
}
.read-cancel,
.read-confirm {
  padding: 6px 14px;
  font-size: 12px;
  font-weight: 600;
  border-radius: 4px;
  cursor: pointer;
  border: 1px solid transparent;
}
.read-cancel {
  background: transparent;
  border-color: var(--border-strong);
  color: var(--text-primary);
}
.read-cancel:hover {
  background: var(--bg-input);
}
.read-confirm {
  background: var(--accent-primary);
  color: var(--bg-base);
}
.read-confirm:hover {
  background: var(--accent-lavender);
}
</style>
