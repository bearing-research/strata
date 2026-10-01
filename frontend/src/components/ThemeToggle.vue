<script setup lang="ts">
import { computed } from 'vue'
import { useTheme } from '../composables/useTheme'

const { mode, cycleMode } = useTheme()

// The icon shows the current mode; the title names the next one.
const icon = computed(() => {
  if (mode.value === 'system') return '◐'
  if (mode.value === 'light') return '☀'
  return '☾'
})

const label = computed(() => {
  if (mode.value === 'system') return 'Theme: system'
  if (mode.value === 'light') return 'Theme: light'
  return 'Theme: dark'
})

const nextLabel = computed(() => {
  if (mode.value === 'system') return 'Switch to light mode'
  if (mode.value === 'light') return 'Switch to dark mode'
  return 'Switch to system preference'
})
</script>

<template>
  <button
    type="button"
    class="theme-toggle"
    :title="`${label} — ${nextLabel}`"
    :aria-label="nextLabel"
    @click="cycleMode"
  >
    <span class="theme-toggle-icon" aria-hidden="true">{{ icon }}</span>
  </button>
</template>

<style scoped>
.theme-toggle {
  background: none;
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 14px;
  line-height: 1;
  cursor: pointer;
  padding: 4px 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition:
    color 120ms ease,
    border-color 120ms ease;
}
.theme-toggle:hover {
  color: var(--text-primary);
  border-color: var(--accent-primary);
}
.theme-toggle:focus-visible {
  outline: none;
  box-shadow: 0 0 0 2px var(--ring-focus);
}
.theme-toggle-icon {
  font-size: 14px;
  width: 16px;
  text-align: center;
}
</style>
