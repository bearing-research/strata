<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useNotebook } from '../stores/notebook'
import { promotableOutputs } from '../utils/artifactRef'

const props = defineProps<{ cellId: string }>()

const {
  registryEnabled,
  registryArtifactsByCell,
  registryNames,
  setAliasAction,
  promoteToTeamAction,
  teamStoreConfigured,
  cellMap,
  pushToast,
  openLineageAction,
} = useNotebook()

interface StripRow {
  key: string
  name: string
  artifactId: string
  version: number
  aliases: Record<string, number>
  tags: Record<string, string>
}

// A cell's published artifacts (GET …/artifacts gives name + tags) joined to
// the registry summary for alias chips (★champion / candidate).
const rows = computed<StripRow[]>(() => {
  const published = registryArtifactsByCell.value[props.cellId] || []
  const aliasesByName: Record<string, Record<string, number>> = {}
  for (const n of registryNames.value) aliasesByName[n.name] = n.aliases
  const out: StripRow[] = []
  for (const art of published) {
    for (const name of art.names) {
      out.push({
        key: `${name}:${art.artifact_id}:${art.version}`,
        name,
        artifactId: art.artifact_id,
        version: art.version,
        aliases: aliasesByName[name] || {},
        tags: Object.fromEntries(Object.entries(art.tags).filter(([k]) => !k.startsWith('nb_'))),
      })
    }
  }
  return out
})

const openMenu = ref<string | null>(null)
const busy = ref<string | null>(null)

async function promote(row: StripRow, alias: 'champion' | 'candidate') {
  openMenu.value = null
  busy.value = row.key
  try {
    const result = await setAliasAction(row.name, alias, row.artifactId, row.version)
    if (result.status === 'pending') pushToast(`⏳ ${alias} change pending approval`, 'info')
    else if (result.status === 'unchanged') pushToast(`${row.name} → ${alias} (no change)`, 'info')
    else pushToast(`✓ ${row.name} → ${alias}`, 'success')
  } catch (err) {
    pushToast(err instanceof Error ? err.message : `Failed to set ${alias}`, 'error')
  } finally {
    busy.value = null
  }
}

// Any stored output of the cell can go to the team store, not only a result the
// cell published itself with put(name=...). Offered only when a team store is
// configured; without one the promote route has nowhere to send it.
// Only a ready cell, and only what it still defines: the backend's map keeps
// every variable a cell has ever stored, so after a rename or an unrun edit it
// would otherwise offer an outdated result for promotion under a team name.
const shareable = computed<Record<string, string>>(() => {
  if (!teamStoreConfigured.value) return {}
  return promotableOutputs(cellMap.value.get(props.cellId))
})
const shareVariables = computed(() => Object.keys(shareable.value).sort())

const sharing = ref(false)
const shareVariable = ref('')
const shareName = ref('')
// Whether the person typed a name. Until they do, the name follows the chosen
// output, so switching from `model` to `scaler` cannot promote the scaler as
// "model" over the team's real model.
const shareNameEdited = ref(false)
const shareBusy = ref(false)

function openShare() {
  sharing.value = true
  if (!shareVariables.value.includes(shareVariable.value)) {
    shareVariable.value = shareVariables.value[0] || ''
  }
  shareNameEdited.value = false
  shareName.value = shareVariable.value
}

watch(shareVariable, (variable) => {
  if (!shareNameEdited.value) shareName.value = variable
})

async function share() {
  const uri = shareable.value[shareVariable.value]
  const name = shareName.value.trim()
  if (!uri || !name) return
  shareBusy.value = true
  try {
    const result = await promoteToTeamAction(props.cellId, uri, name)
    const carried = result.copied === 1 ? '1 artifact' : `${result.copied} artifacts`
    pushToast(`✓ ${result.name} in the team store (${carried} copied)`, 'success')
    sharing.value = false
  } catch (err) {
    pushToast(err instanceof Error ? err.message : `Failed to promote ${name}`, 'error')
  } finally {
    shareBusy.value = false
  }
}

function tagList(tags: Record<string, string>): string {
  return Object.entries(tags)
    .map(([k, v]) => `${k}=${v}`)
    .join('  ')
}
</script>

<template>
  <div v-if="registryEnabled && (rows.length || shareVariables.length)" class="cell-artifact-strip">
    <div v-for="row in rows" :key="row.key" class="strip-row">
      <span class="glyph">⬡</span>
      <span class="name">{{ row.name }}</span>
      <span class="ver">v{{ row.version }}</span>
      <span
        v-for="(ver, alias) in row.aliases"
        :key="alias"
        class="chip"
        :class="{ champ: alias === 'champion' }"
        >{{ alias === 'champion' ? '★' : '' }}{{ alias }}=v{{ ver }}</span
      >
      <span v-if="tagList(row.tags)" class="tags">{{ tagList(row.tags) }}</span>
      <span class="spacer"></span>
      <div class="promote-wrap">
        <button
          class="promote-btn"
          :disabled="busy === row.key"
          @click="openMenu = openMenu === row.key ? null : row.key"
        >
          Promote ▾
        </button>
        <div v-if="openMenu === row.key" class="promote-menu">
          <button @click="promote(row, 'champion')">Set as champion</button>
          <button @click="promote(row, 'candidate')">Set as candidate</button>
        </div>
      </div>
      <button
        class="lineage-btn"
        title="View lineage"
        @click="openLineageAction(row.artifactId, row.version, row.name)"
      >
        ⎘
      </button>
    </div>
    <div v-if="shareVariables.length" class="strip-row share-row">
      <template v-if="sharing">
        <span class="glyph">⇪</span>
        <select
          v-if="shareVariables.length > 1"
          v-model="shareVariable"
          class="share-input"
          aria-label="Output to promote"
        >
          <option v-for="v in shareVariables" :key="v" :value="v">{{ v }}</option>
        </select>
        <span v-else class="name">{{ shareVariable }}</span>
        <span>as</span>
        <input
          v-model="shareName"
          class="share-input share-name"
          placeholder="team/name"
          aria-label="Team name"
          @input="shareNameEdited = true"
          @keydown.enter="share"
          @keydown.esc="sharing = false"
        />
        <span class="spacer"></span>
        <button class="promote-btn" :disabled="shareBusy || !shareName.trim()" @click="share">
          {{ shareBusy ? 'Promoting…' : 'Promote' }}
        </button>
        <button class="promote-btn" :disabled="shareBusy" @click="sharing = false">Cancel</button>
      </template>
      <template v-else>
        <span class="spacer"></span>
        <button
          class="share-open"
          title="Copy this output and everything behind it to the team store, under a name"
          @click="openShare"
        >
          ⇪ Promote to team…
        </button>
      </template>
    </div>
  </div>
</template>

<style scoped>
.cell-artifact-strip {
  margin-top: 4px;
  font-size: 12px;
}
.strip-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 8px;
  border-top: 1px solid var(--border-subtle);
  color: var(--text-muted);
}
.glyph {
  color: var(--accent-primary, #3b82f6);
}
.name {
  font-weight: 600;
  color: var(--text-primary);
}
.chip {
  display: inline-block;
  padding: 0 6px;
  border-radius: 10px;
  background: var(--cat-surface1, var(--border-subtle));
  font-size: 11px;
}
.chip.champ {
  background: var(--tint-success, #e6f4ea);
  color: var(--accent-success, #1e7e34);
}
.tags {
  font-family: var(--font-mono, monospace);
  font-size: 11px;
}
.spacer {
  flex: 1;
}
.promote-wrap {
  position: relative;
}
.promote-btn {
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: 1px 8px;
  background: var(--bg-elevated);
  cursor: pointer;
  font-size: 11px;
  color: var(--text-primary);
}
.promote-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
.promote-menu {
  position: absolute;
  right: 0;
  top: 100%;
  z-index: 20;
  background: var(--bg-elevated);
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
  color: var(--text-primary);
}
.promote-menu button:hover {
  background: var(--bg-hover);
}
.share-open {
  border: none;
  background: none;
  cursor: pointer;
  color: var(--text-muted);
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 4px;
}
.share-open:hover {
  color: var(--accent-primary, #3b82f6);
  background: var(--bg-hover);
}
.share-input {
  font-size: 11px;
  padding: 1px 4px;
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  background: var(--bg-elevated);
  color: var(--text-primary);
}
.share-name {
  min-width: 0;
  flex: 0 1 180px;
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
  background: var(--bg-hover);
}
</style>
