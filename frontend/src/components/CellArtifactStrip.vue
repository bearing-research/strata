<script setup lang="ts">
import { computed, ref } from 'vue'
import { useNotebook } from '../stores/notebook'

const props = defineProps<{ cellId: string }>()

const { registryEnabled, registryArtifactsByCell, registryNames, setAliasAction, pushToast } =
  useNotebook()

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

function tagList(tags: Record<string, string>): string {
  return Object.entries(tags)
    .map(([k, v]) => `${k}=${v}`)
    .join('  ')
}
</script>

<template>
  <div v-if="registryEnabled && rows.length" class="cell-artifact-strip">
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
  color: var(--text);
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
  background: var(--bg, #fff);
  cursor: pointer;
  font-size: 11px;
  color: var(--text);
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
</style>
