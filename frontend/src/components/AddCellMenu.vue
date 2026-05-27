<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

interface CellTypeOption {
  language: string
  label: string
  description: string
}

const CELL_TYPES: CellTypeOption[] = [
  { language: 'python', label: 'Python', description: 'Code cell — the default' },
  { language: 'sql', label: 'SQL', description: 'Query a connected database' },
  { language: 'r', label: 'R', description: 'R cell — stats and tidyverse' },
  { language: 'prompt', label: 'Prompt', description: 'LLM-powered template' },
  { language: 'markdown', label: 'Markdown', description: 'Documentation / prose' },
]

defineProps<{
  disabled?: boolean
  variant?: 'header' | 'inline'
}>()

const emit = defineEmits<{
  (e: 'select', language: string): void
}>()

const open = ref(false)
const root = ref<HTMLElement | null>(null)

function toggle() {
  open.value = !open.value
}

function pick(language: string) {
  open.value = false
  emit('select', language)
}

function onDocumentClick(event: MouseEvent) {
  if (!open.value) return
  const el = root.value
  if (el && event.target instanceof Node && !el.contains(event.target)) {
    open.value = false
  }
}

function onKeydown(event: KeyboardEvent) {
  if (open.value && event.key === 'Escape') {
    open.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', onDocumentClick)
  document.addEventListener('keydown', onKeydown)
})
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocumentClick)
  document.removeEventListener('keydown', onKeydown)
})
</script>

<template>
  <div ref="root" class="add-cell-menu" :class="`variant-${variant ?? 'inline'}`">
    <button
      type="button"
      class="add-cell-trigger"
      :class="{ 'is-open': open }"
      :disabled="disabled"
      :aria-haspopup="true"
      :aria-expanded="open"
      data-testid="add-cell-menu-trigger"
      @click="toggle"
    >
      <span class="trigger-label">+ Add cell</span>
      <span class="trigger-caret" aria-hidden="true">▾</span>
    </button>

    <div v-if="open" class="add-cell-dropdown" role="menu" data-testid="add-cell-menu-dropdown">
      <button
        v-for="opt in CELL_TYPES"
        :key="opt.language"
        type="button"
        class="add-cell-option"
        role="menuitem"
        :data-testid="`add-cell-option-${opt.language}`"
        @click="pick(opt.language)"
      >
        <span class="option-label">{{ opt.label }}</span>
        <span class="option-description">{{ opt.description }}</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.add-cell-menu {
  position: relative;
  display: inline-block;
}

.add-cell-trigger {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.75rem;
  font-size: 0.875rem;
  border: 1px solid var(--border-color, #d0d0d0);
  background: var(--bg-secondary, #fff);
  color: var(--text-primary, #222);
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.1s ease;
}

.add-cell-trigger:hover:not(:disabled) {
  background: var(--bg-hover, #f3f3f3);
}

.add-cell-trigger:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.add-cell-trigger.is-open {
  background: var(--bg-hover, #f3f3f3);
}

.trigger-caret {
  font-size: 0.7rem;
  line-height: 1;
}

.variant-inline .add-cell-trigger {
  padding: 0.5rem 0.9rem;
  font-size: 0.95rem;
}

.add-cell-dropdown {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  min-width: 14rem;
  background: var(--bg-primary, #fff);
  border: 1px solid var(--border-color, #d0d0d0);
  border-radius: 6px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  z-index: 50;
  padding: 0.25rem 0;
  display: flex;
  flex-direction: column;
}

.add-cell-option {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.1rem;
  padding: 0.5rem 0.75rem;
  background: transparent;
  border: 0;
  cursor: pointer;
  text-align: left;
  color: var(--text-primary, #222);
}

.add-cell-option:hover {
  background: var(--bg-hover, #f1f4f8);
}

.option-label {
  font-size: 0.9rem;
  font-weight: 500;
}

.option-description {
  font-size: 0.75rem;
  color: var(--text-secondary, #666);
}
</style>
