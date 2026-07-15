<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

interface FormatOption {
  format: 'markdown' | 'html'
  appView?: boolean
  label: string
  description: string
}

const FORMATS: FormatOption[] = [
  {
    format: 'markdown',
    label: 'Markdown',
    description: 'Drops into GitHub, mkdocs, Confluence',
  },
  {
    format: 'html',
    label: 'HTML',
    description: 'Standalone file, syntax-highlighted',
  },
  {
    format: 'html',
    appView: true,
    label: 'App snapshot',
    description: 'Widgets + outputs only, no code — a portable frozen dashboard',
  },
]

defineProps<{
  disabled?: boolean
}>()

const emit = defineEmits<{
  (e: 'select', format: 'markdown' | 'html', appView: boolean): void
}>()

const open = ref(false)
const root = ref<HTMLElement | null>(null)

function toggle() {
  open.value = !open.value
}

function pick(opt: FormatOption) {
  open.value = false
  emit('select', opt.format, Boolean(opt.appView))
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
  <div ref="root" class="export-menu">
    <button
      type="button"
      class="export-trigger"
      :class="{ 'is-open': open }"
      :disabled="disabled"
      :aria-haspopup="true"
      :aria-expanded="open"
      data-testid="export-menu-trigger"
      title="Export this notebook as a single file"
      @click="toggle"
    >
      <span class="trigger-label">Export</span>
      <span class="trigger-caret" aria-hidden="true">▾</span>
    </button>

    <div v-if="open" class="export-dropdown" role="menu" data-testid="export-menu-dropdown">
      <button
        v-for="opt in FORMATS"
        :key="opt.appView ? `${opt.format}-app` : opt.format"
        type="button"
        class="export-option"
        role="menuitem"
        :data-testid="`export-option-${opt.appView ? 'snapshot' : opt.format}`"
        @click="pick(opt)"
      >
        <span class="option-label">{{ opt.label }}</span>
        <span class="option-description">{{ opt.description }}</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.export-menu {
  position: relative;
  display: inline-block;
}

.export-trigger {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 6px 12px;
  font-size: 13px;
  border: 1px solid var(--border);
  background: var(--bg-input);
  color: var(--text-primary);
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.1s ease;
}

.export-trigger:hover:not(:disabled) {
  background: var(--bg-hover);
}

.export-trigger:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.export-trigger.is-open {
  background: var(--bg-hover);
}

.trigger-caret {
  font-size: 0.7rem;
  line-height: 1;
}

.export-dropdown {
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  min-width: 16rem;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 6px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  z-index: 50;
  padding: 0.25rem 0;
  display: flex;
  flex-direction: column;
}

.export-option {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.15rem;
  padding: 0.5rem 0.75rem;
  border: none;
  background: transparent;
  text-align: left;
  cursor: pointer;
  font: inherit;
  color: inherit;
}

.export-option:hover {
  background: var(--bg-hover);
}

.option-label {
  font-weight: 500;
  font-size: 0.9rem;
}

.option-description {
  font-size: 0.75rem;
  color: var(--text-muted);
}
</style>
