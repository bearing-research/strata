<script setup lang="ts">
import { computed, ref } from 'vue'
import { useNotebook } from '../stores/notebook'

interface SchemaColumn {
  name: string
  type: string
  nullable: boolean | null
}

interface SchemaTable {
  catalog: string | null
  schema: string | null
  name: string
  columns: SchemaColumn[]
}

interface ConnectionSchemaState {
  status: 'idle' | 'loading' | 'ready' | 'error'
  tables: SchemaTable[]
  error: string | null
  expanded: boolean
  expandedTables: Set<string>
}

const { notebook, connected, getConnectionSchemaAction } = useNotebook()
const showPanel = ref(false)
const states = ref<Record<string, ConnectionSchemaState>>({})

const connectionCount = computed(() => notebook.connections.length)

function getState(name: string): ConnectionSchemaState {
  if (!states.value[name]) {
    states.value[name] = {
      status: 'idle',
      tables: [],
      error: null,
      expanded: false,
      expandedTables: new Set(),
    }
  }
  return states.value[name]
}

async function loadSchema(name: string, force = false) {
  const state = getState(name)
  if (state.status === 'loading') return
  if (state.status === 'ready' && !force) return

  state.status = 'loading'
  state.error = null
  try {
    const res = await getConnectionSchemaAction(name)
    state.tables = (res.tables || []).map((t) => ({
      catalog: t.catalog,
      schema: t.schema,
      name: t.name,
      columns: (t.columns || []).map((c) => ({
        name: c.name,
        type: c.type,
        nullable: c.nullable,
      })),
    }))
    state.status = 'ready'
  } catch (err) {
    state.status = 'error'
    state.error = err instanceof Error ? err.message : String(err)
  }
}

async function toggleConnection(name: string) {
  const state = getState(name)
  state.expanded = !state.expanded
  if (state.expanded && state.status === 'idle') {
    await loadSchema(name)
  }
}

function toggleTable(name: string, tableKey: string) {
  const state = getState(name)
  if (state.expandedTables.has(tableKey)) {
    state.expandedTables.delete(tableKey)
  } else {
    state.expandedTables.add(tableKey)
  }
  // Trigger reactivity on the Set.
  state.expandedTables = new Set(state.expandedTables)
}

function tableKey(t: SchemaTable): string {
  return [t.catalog, t.schema, t.name].filter(Boolean).join('.')
}

function nullabilityLabel(col: SchemaColumn): string {
  if (col.nullable === null) return ''
  return col.nullable ? '' : ' NOT NULL'
}
</script>

<template>
  <div class="schema-panel">
    <button class="schema-toggle" @click="showPanel = !showPanel">
      Schema
      <span class="schema-count">{{ connectionCount }}</span>
      <span class="toggle-icon">{{ showPanel ? '&#9650;' : '&#9660;' }}</span>
    </button>

    <div v-if="showPanel" class="schema-content">
      <p v-if="connectionCount === 0" class="schema-empty">
        Declare a connection in the Connections panel to browse its tables.
      </p>

      <div v-for="conn in notebook.connections" :key="conn.name" class="conn-block">
        <div class="conn-header">
          <button class="conn-toggle" :disabled="!connected" @click="toggleConnection(conn.name)">
            <span class="caret">{{ getState(conn.name).expanded ? '▾' : '▸' }}</span>
            <span class="conn-name">{{ conn.name }}</span>
            <span class="conn-driver">{{ conn.driver }}</span>
          </button>
          <button
            v-if="getState(conn.name).expanded"
            class="conn-refresh"
            :disabled="!connected || getState(conn.name).status === 'loading'"
            title="Re-fetch schema"
            @click="loadSchema(conn.name, true)"
          >
            ↻
          </button>
        </div>

        <div v-if="getState(conn.name).expanded" class="conn-body">
          <div v-if="getState(conn.name).status === 'loading'" class="schema-status">Loading…</div>
          <div v-else-if="getState(conn.name).status === 'error'" class="schema-error">
            {{ getState(conn.name).error }}
          </div>
          <div
            v-else-if="
              getState(conn.name).status === 'ready' && getState(conn.name).tables.length === 0
            "
            class="schema-status"
          >
            (no tables)
          </div>
          <ul v-else-if="getState(conn.name).status === 'ready'" class="table-list">
            <li v-for="t in getState(conn.name).tables" :key="tableKey(t)" class="table-item">
              <button class="table-toggle" @click="toggleTable(conn.name, tableKey(t))">
                <span class="caret">
                  {{ getState(conn.name).expandedTables.has(tableKey(t)) ? '▾' : '▸' }}
                </span>
                <span class="table-name">{{ t.name }}</span>
                <span v-if="t.schema" class="table-qualifier">{{ t.schema }}</span>
                <span class="table-col-count">{{ t.columns.length }} cols</span>
              </button>
              <ul v-if="getState(conn.name).expandedTables.has(tableKey(t))" class="column-list">
                <li v-for="col in t.columns" :key="col.name" class="column-item">
                  <span class="column-name">{{ col.name }}</span>
                  <span class="column-type">{{ col.type }}{{ nullabilityLabel(col) }}</span>
                </li>
              </ul>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.schema-panel {
  margin-top: 12px;
  border-top: 1px solid var(--border-subtle);
  padding-top: 8px;
}

.schema-toggle {
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

.schema-toggle:hover {
  color: var(--text-primary);
}

.schema-count {
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

.schema-content {
  margin-top: 8px;
}

.schema-empty {
  font-size: 12px;
  color: var(--text-muted);
  font-style: italic;
}

.conn-block {
  margin-bottom: 6px;
}

.conn-header {
  display: flex;
  align-items: center;
  gap: 4px;
}

.conn-toggle {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 6px;
  background: none;
  border: 1px solid transparent;
  color: var(--text-primary);
  font-size: 13px;
  cursor: pointer;
  padding: 4px 6px;
  text-align: left;
  border-radius: 4px;
}

.conn-toggle:hover:not(:disabled) {
  background: var(--bg-hover, rgba(0, 0, 0, 0.04));
}

.conn-name {
  font-weight: 600;
}

.conn-driver {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-muted);
  font-family: var(--font-mono, monospace);
}

.conn-refresh {
  background: none;
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  cursor: pointer;
  padding: 2px 8px;
  color: var(--text-secondary);
  font-size: 13px;
}

.conn-refresh:hover:not(:disabled) {
  color: var(--text-primary);
}

.conn-body {
  padding-left: 16px;
}

.caret {
  font-size: 10px;
  width: 10px;
  display: inline-block;
}

.schema-status,
.schema-error {
  font-size: 12px;
  padding: 4px 6px;
}

.schema-error {
  color: var(--accent-danger);
  background: var(--tint-danger);
  border-radius: 4px;
}

.table-list,
.column-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.table-item {
  margin: 1px 0;
}

.table-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  background: none;
  border: 0;
  cursor: pointer;
  padding: 2px 6px;
  text-align: left;
  font-size: 12px;
  color: var(--text-primary);
  border-radius: 4px;
}

.table-toggle:hover {
  background: var(--bg-hover, rgba(0, 0, 0, 0.04));
}

.table-name {
  font-family: var(--font-mono, monospace);
}

.table-qualifier {
  font-size: 11px;
  color: var(--text-muted);
}

.table-col-count {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-muted);
}

.column-list {
  margin-left: 22px;
  border-left: 1px solid var(--border-subtle);
}

.column-item {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 1px 8px;
  font-size: 12px;
  font-family: var(--font-mono, monospace);
}

.column-name {
  color: var(--text-primary);
}

.column-type {
  color: var(--text-muted);
  font-size: 11px;
}
</style>
