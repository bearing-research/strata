<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

const open = ref(false)
const root = ref<HTMLElement | null>(null)

function onDocumentClick(event: MouseEvent) {
  const el = root.value
  if (open.value && el && event.target instanceof Node && !el.contains(event.target)) {
    open.value = false
  }
}

function onKeydown(event: KeyboardEvent) {
  if (open.value && event.key === 'Escape') open.value = false
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
  <div ref="root" class="overflow-menu">
    <button
      type="button"
      class="overflow-trigger"
      :class="{ 'is-open': open }"
      aria-label="More actions"
      aria-haspopup="true"
      :aria-expanded="open"
      data-testid="notebook-overflow-trigger"
      title="More actions"
      @click="open = !open"
    >
      &#x22EF;
    </button>
    <div v-if="open" class="overflow-dropdown" role="menu" @click="open = false">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.overflow-menu {
  position: relative;
  display: inline-block;
}

.overflow-trigger {
  padding: 6px 10px;
  font-size: 13px;
  line-height: 1.2;
  border: 1px solid var(--border);
  background: var(--bg-input);
  color: var(--text-primary);
  border-radius: 6px;
  cursor: pointer;
}

.overflow-trigger:hover,
.overflow-trigger.is-open {
  background: var(--bg-hover);
}

.overflow-dropdown {
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  min-width: 11rem;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 6px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  z-index: 50;
  padding: 0.25rem 0;
  display: flex;
  flex-direction: column;
}

.overflow-dropdown :slotted(.overflow-item) {
  display: block;
  width: 100%;
  padding: 0.45rem 0.75rem;
  border: none;
  background: transparent;
  color: var(--text-primary);
  font: inherit;
  font-size: 13px;
  line-height: 1.4;
  text-align: left;
  text-decoration: none;
  white-space: nowrap;
  cursor: pointer;
}

.overflow-dropdown :slotted(.overflow-item:hover:not(:disabled)) {
  background: var(--bg-hover);
}

.overflow-dropdown :slotted(.overflow-item:disabled) {
  opacity: 0.5;
  cursor: not-allowed;
}

.overflow-dropdown :slotted(.overflow-item-danger) {
  border-top: 1px solid var(--border-subtle);
  color: var(--accent-danger);
}
</style>
