<script setup lang="ts">
import { computed, ref } from 'vue'
import { useNotebook } from '../stores/notebook'
import ConnectionListEditor from './ConnectionListEditor.vue'

const { notebook, connected, updateNotebookConnectionsAction } = useNotebook()
const showPanel = ref(false)
const lastError = ref<string | null>(null)

const connectionCount = computed(() => notebook.connections.length)

async function save(connections: typeof notebook.connections) {
  lastError.value = null
  try {
    await updateNotebookConnectionsAction(connections)
  } catch (err) {
    lastError.value = err instanceof Error ? err.message : String(err)
    throw err
  }
}
</script>

<template>
  <div class="connections-panel">
    <button class="connections-toggle" @click="showPanel = !showPanel">
      Connections
      <span class="connection-count">{{ connectionCount }}</span>
      <span class="toggle-icon">{{ showPanel ? '&#9650;' : '&#9660;' }}</span>
    </button>

    <div v-if="showPanel" class="connections-content">
      <p class="connections-copy">
        SQL connections referenced by ``# @sql connection=&lt;name&gt;``. Use
        <code>$&#123;VAR&#125;</code> for credentials — literal secrets are blanked when
        notebook.toml is saved.
      </p>
      <ConnectionListEditor
        :connections="notebook.connections"
        :read-only="!connected"
        @save="save"
      />
      <p v-if="lastError" class="connections-error">{{ lastError }}</p>
    </div>
  </div>
</template>

<style scoped>
.connections-panel {
  margin-top: 12px;
  border-top: 1px solid var(--border-subtle);
  padding-top: 8px;
}

.connections-toggle {
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

.connections-toggle:hover {
  color: var(--text-primary);
}

.connection-count {
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

.connections-content {
  margin-top: 8px;
}

.connections-copy {
  margin-bottom: 8px;
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.4;
}

.connections-copy code {
  background: var(--bg-input);
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 11px;
}

.connections-error {
  margin-top: 8px;
  padding: 6px 8px;
  background: var(--tint-danger);
  color: var(--accent-danger);
  border-radius: 4px;
  font-size: 12px;
}
</style>
