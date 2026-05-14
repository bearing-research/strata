<script setup lang="ts">
import { ref, watch } from 'vue'

const props = defineProps<{
  /** Whether the modal is open. Parent owns the open/closed state. */
  open: boolean
  /** Versions the deployment allows (typically discovered via uv). */
  available: readonly string[]
  /** Currently-declared Python minor in pyproject.toml. */
  current: string
  /** Disable the Confirm button while the PUT is in-flight. */
  busy?: boolean
  /** Server-side error text from the last failed attempt. */
  errorMessage?: string | null
}>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'confirm', version: string): void
}>()

const selected = ref(props.current)

// Reset the selected version whenever the modal re-opens so a previous
// pick doesn't leak across open/close cycles.
watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      selected.value = props.current
    }
  },
)

function onConfirm() {
  if (props.busy) return
  emit('confirm', selected.value)
}
</script>

<template>
  <div v-if="open" class="pyver-overlay" role="dialog" aria-modal="true">
    <div class="pyver-card">
      <div class="pyver-title">Change notebook Python version</div>
      <div class="pyver-body">
        Switching the notebook to a different Python minor rebuilds the
        virtual environment from scratch and invalidates every cached cell
        output. Cell sources stay intact and re-run on the new interpreter
        the next time you execute them.
      </div>

      <label class="pyver-label">
        Python version
        <select v-model="selected" class="pyver-select" :disabled="busy">
          <option v-for="version in available" :key="version" :value="version">
            {{ version }}{{ version === current ? ' (current)' : '' }}
          </option>
        </select>
      </label>

      <div v-if="errorMessage" class="pyver-error">{{ errorMessage }}</div>

      <div class="pyver-actions">
        <button class="pyver-cancel" :disabled="busy" @click="emit('close')">Cancel</button>
        <button
          class="pyver-confirm"
          :disabled="busy || selected === current"
          @click="onConfirm"
        >
          {{ busy ? 'Rebuilding…' : `Change to ${selected}` }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.pyver-overlay {
  position: fixed;
  inset: 0;
  background: rgba(10, 10, 20, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.pyver-card {
  background: var(--bg-elevated);
  border: 1px solid var(--accent-primary);
  border-radius: 8px;
  padding: 18px 20px;
  min-width: 360px;
  max-width: 480px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
}
.pyver-title {
  color: var(--text-primary);
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 8px;
}
.pyver-body {
  color: var(--text-secondary);
  font-size: 13px;
  line-height: 1.45;
  margin-bottom: 14px;
}
.pyver-label {
  display: block;
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 12px;
}
.pyver-select {
  display: block;
  margin-top: 4px;
  width: 100%;
  padding: 6px 8px;
  font-size: 13px;
  background: var(--bg-input);
  color: var(--text-primary);
  border: 1px solid var(--border-strong);
  border-radius: 4px;
}
.pyver-select:disabled {
  opacity: 0.5;
}
.pyver-error {
  color: var(--accent-danger);
  font-size: 12px;
  margin-bottom: 12px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.pyver-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
.pyver-cancel,
.pyver-confirm {
  padding: 6px 14px;
  font-size: 12px;
  font-weight: 600;
  border-radius: 4px;
  cursor: pointer;
  border: 1px solid transparent;
}
.pyver-cancel {
  background: transparent;
  border-color: var(--border-strong);
  color: var(--text-primary);
}
.pyver-cancel:hover:not(:disabled) {
  background: var(--bg-input);
}
.pyver-confirm {
  background: var(--accent-primary);
  color: var(--bg-base);
}
.pyver-confirm:hover:not(:disabled) {
  background: var(--accent-lavender);
}
.pyver-cancel:disabled,
.pyver-confirm:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
