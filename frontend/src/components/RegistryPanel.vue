<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useNotebook } from '../stores/notebook'

const {
  registryNames,
  registryPending,
  registryAudit,
  registryLoading,
  registryError,
  refreshRegistryAction,
  fetchRegistryAuditAction,
  setAliasAction,
  approvePendingAction,
  rejectPendingAction,
  openLineageAction,
  pushToast,
} = useNotebook()

const auditOpen = ref(false)
const busy = ref<string | null>(null) // key of the row currently mutating
const openMenu = ref<string | null>(null) // name whose Promote menu is open

onMounted(() => {
  void refreshRegistryAction()
  void fetchRegistryAuditAction()
})

async function promote(
  name: string,
  alias: 'champion' | 'candidate',
  artifactId: string,
  version: number,
) {
  openMenu.value = null
  busy.value = `${name}:${alias}`
  try {
    const result = await setAliasAction(name, alias, artifactId, version)
    if (result.status === 'pending') {
      pushToast(`⏳ ${alias} change pending approval`, 'info')
    } else if (result.status === 'unchanged') {
      pushToast(`${name} → ${alias} (no change)`, 'info')
    } else {
      pushToast(`✓ ${name} → ${alias}`, 'success')
    }
  } catch (err) {
    pushToast(err instanceof Error ? err.message : `Failed to set ${alias}`, 'error')
  } finally {
    busy.value = null
  }
}

async function approve(name: string, alias: string) {
  busy.value = `pending:${name}:${alias}`
  try {
    await approvePendingAction(name, alias)
    pushToast(`✓ approved ${name}@${alias}`, 'success')
  } catch (err) {
    pushToast(err instanceof Error ? err.message : 'Approve failed', 'error')
  } finally {
    busy.value = null
  }
}

async function reject(name: string, alias: string) {
  busy.value = `pending:${name}:${alias}`
  try {
    await rejectPendingAction(name, alias)
    pushToast(`rejected ${name}@${alias}`, 'info')
  } catch (err) {
    pushToast(err instanceof Error ? err.message : 'Reject failed', 'error')
  } finally {
    busy.value = null
  }
}

function fmtTime(epoch: number): string {
  if (!epoch) return ''
  return new Date(epoch * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function tagList(tags: Record<string, string>): string {
  return Object.entries(tags)
    .map(([k, v]) => `${k}=${v}`)
    .join('  ')
}
</script>

<template>
  <div class="registry-panel">
    <div v-if="registryError" class="registry-error">{{ registryError }}</div>

    <!-- Pending approvals (action items) -->
    <section v-if="registryPending.length" class="pending">
      <div class="section-label">⏳ Pending</div>
      <div v-for="p in registryPending" :key="`${p.name}:${p.alias}`" class="pending-row">
        <span class="pending-target"
          >{{ p.name }} <span class="chip alias">@{{ p.alias }}</span> → v{{ p.version }}</span
        >
        <span class="pending-actions">
          <button
            class="btn approve"
            :disabled="busy === `pending:${p.name}:${p.alias}`"
            @click="approve(p.name, p.alias)"
          >
            Approve
          </button>
          <button
            class="btn reject"
            :disabled="busy === `pending:${p.name}:${p.alias}`"
            @click="reject(p.name, p.alias)"
          >
            Reject
          </button>
        </span>
      </div>
    </section>

    <!-- Registry state -->
    <section class="names">
      <div v-if="registryLoading && !registryNames.length" class="empty">Loading…</div>
      <div v-else-if="!registryNames.length" class="empty">
        No published artifacts yet — call <code>strata.put(…, name=…)</code> in a cell.
      </div>
      <table v-else class="names-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Aliases</th>
            <th>Latest</th>
            <th>Tags</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="n in registryNames" :key="n.name">
            <td class="name">{{ n.name }}</td>
            <td class="aliases">
              <span
                v-for="(ver, alias) in n.aliases"
                :key="alias"
                class="chip"
                :class="{ champ: alias === 'champion' }"
                >{{ alias === 'champion' ? '★' : '' }}{{ alias }}=v{{ ver }}</span
              >
              <span v-if="!Object.keys(n.aliases).length" class="muted">—</span>
            </td>
            <td class="latest">v{{ n.version }}</td>
            <td class="tags">{{ tagList(n.tags) || '—' }}</td>
            <td class="promote">
              <div class="promote-wrap">
                <button
                  class="btn promote-btn"
                  :disabled="busy?.startsWith(`${n.name}:`)"
                  @click="openMenu = openMenu === n.name ? null : n.name"
                >
                  Promote ▾
                </button>
                <div v-if="openMenu === n.name" class="promote-menu">
                  <button @click="promote(n.name, 'champion', n.artifact_id, n.version)">
                    Set as champion
                  </button>
                  <button @click="promote(n.name, 'candidate', n.artifact_id, n.version)">
                    Set as candidate
                  </button>
                </div>
                <button
                  class="lineage-btn"
                  title="View lineage"
                  @click="openLineageAction(n.artifact_id, n.version, n.name)"
                >
                  ⎘
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </section>

    <!-- Audit (collapsed) -->
    <section class="audit">
      <button class="section-label toggle" @click="auditOpen = !auditOpen">
        {{ auditOpen ? '▾' : '▸' }} Audit ({{ registryAudit.length }})
      </button>
      <div v-if="auditOpen" class="audit-rows">
        <div v-for="(e, i) in registryAudit" :key="i" class="audit-row">
          <span class="audit-time">{{ fmtTime(e.at) }}</span>
          <span class="audit-action">{{ e.action }}</span>
          <span class="audit-target">
            {{ e.name }}<template v-if="e.alias">@{{ e.alias }}</template
            ><template v-if="e.version"> → v{{ e.version }}</template>
          </span>
          <span v-if="e.actor" class="audit-actor">[{{ e.actor }}]</span>
        </div>
        <div v-if="!registryAudit.length" class="empty">No history yet.</div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.registry-panel {
  padding: 8px 10px;
  font-size: 12px;
  overflow: auto;
  height: 100%;
}
.registry-error {
  color: var(--accent-danger, #c0392b);
  margin-bottom: 6px;
}
.section-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
  margin: 6px 0 4px;
}
.pending {
  border: 1px solid var(--accent-warning, #d4a017);
  border-radius: 6px;
  padding: 6px 8px;
  margin-bottom: 8px;
}
.pending-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  padding: 2px 0;
}
.pending-actions {
  display: flex;
  gap: 6px;
}
.names-table {
  width: 100%;
  border-collapse: collapse;
}
.names-table th {
  text-align: left;
  font-weight: 600;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border-subtle);
  padding: 3px 6px;
}
.names-table td {
  padding: 3px 6px;
  border-bottom: 1px solid var(--border-subtle);
  vertical-align: middle;
}
.name {
  font-weight: 600;
  color: var(--text);
}
.chip {
  display: inline-block;
  padding: 1px 6px;
  margin-right: 4px;
  border-radius: 10px;
  background: var(--cat-surface1, var(--border-subtle));
  font-size: 11px;
}
.chip.champ {
  background: var(--tint-success, #e6f4ea);
  color: var(--accent-success, #1e7e34);
}
.chip.alias {
  background: var(--tint-primary, #e8f0fe);
}
.muted,
.tags {
  color: var(--text-muted);
}
.promote-wrap {
  position: relative;
}
.promote-menu {
  position: absolute;
  right: 0;
  top: 100%;
  z-index: 20;
  background: var(--bg, #fff);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  min-width: 140px;
}
.promote-menu button {
  text-align: left;
  padding: 6px 10px;
  background: none;
  border: none;
  cursor: pointer;
  color: var(--text);
}
.promote-menu button:hover {
  background: var(--bg-subtle);
}
.promote {
  display: flex;
  align-items: center;
  gap: 4px;
  justify-content: flex-end;
}
.lineage-btn {
  border: none;
  background: none;
  cursor: pointer;
  color: var(--text-muted);
  font-size: 18px;
  line-height: 1;
  padding: 3px 7px;
  border-radius: 4px;
}
.lineage-btn:hover {
  color: var(--accent-primary, #3b82f6);
  background: var(--bg-subtle);
}
.btn {
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: 2px 8px;
  background: var(--bg, #fff);
  cursor: pointer;
  font-size: 11px;
  color: var(--text);
}
.btn:disabled {
  opacity: 0.5;
  cursor: default;
}
.btn.approve {
  border-color: var(--accent-success, #1e7e34);
}
.btn.reject {
  border-color: var(--accent-danger, #c0392b);
}
.toggle {
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
}
.audit-rows {
  font-family: var(--font-mono, monospace);
  font-size: 11px;
}
.audit-row {
  display: flex;
  gap: 6px;
  padding: 1px 0;
  color: var(--text-muted);
}
.audit-time {
  color: var(--text);
  min-width: 42px;
}
.empty {
  color: var(--text-muted);
  padding: 8px 0;
}
.empty code {
  font-family: var(--font-mono, monospace);
}
</style>
