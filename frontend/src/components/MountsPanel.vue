<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useNotebook } from '../stores/notebook'
import MountListEditor from './MountListEditor.vue'

const props = defineProps<{ attention?: boolean }>()

const { notebook, connected, updateNotebookMountsAction } = useNotebook()
const showPanel = ref(false)
watch(
  () => props.attention,
  (needed) => {
    if (needed) showPanel.value = true
  },
  { immediate: true },
)

const mountCount = computed(() => notebook.mounts.length)
</script>

<template>
  <div class="mounts-panel">
    <button class="mounts-toggle" @click="showPanel = !showPanel">
      Mounts
      <span class="mount-count">{{ mountCount }}</span>
      <span class="toggle-icon">{{ showPanel ? '&#9650;' : '&#9660;' }}</span>
    </button>

    <div v-if="showPanel" class="mounts-content">
      <p class="mounts-copy">Notebook defaults. Cell overrides can replace these per cell.</p>
      <MountListEditor
        :mounts="notebook.mounts"
        title="Notebook Defaults"
        :read-only="!connected"
        @save="updateNotebookMountsAction"
      />
    </div>
  </div>
</template>

<style scoped>
.mounts-panel {
  margin-top: 12px;
  border-top: 1px solid var(--border-subtle);
  padding-top: 8px;
}

.mounts-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  background: none;
  border: none;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  cursor: pointer;
  padding: 4px 0;
}

.mounts-toggle:hover {
  color: var(--text-primary);
}

.mount-count {
  background: var(--bg-input);
  color: var(--accent-primary);
  padding: 1px 6px;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
}

.toggle-icon {
  margin-left: auto;
  font-size: 10px;
}

.mounts-content {
  margin-top: 8px;
}

.mounts-copy {
  margin-bottom: 8px;
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.4;
}
</style>
