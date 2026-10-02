<script setup lang="ts">
import { ref, watch } from 'vue'

const _SENSITIVE_PATTERNS = ['KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'CREDENTIAL']

function isSensitiveKey(key: string): boolean {
  const upper = key.toUpperCase()
  return _SENSITIVE_PATTERNS.some((p) => upper.includes(p))
}

interface EnvRow {
  localId: string
  key: string
  value: string
}

const props = withDefaults(
  defineProps<{
    env: Record<string, string>
    envSources?: Record<string, string>
    title?: string
    compact?: boolean
    readOnly?: boolean
  }>(),
  {
    title: 'Environment Variables',
    envSources: () => ({}),
    compact: false,
    readOnly: false,
  },
)

function sourceLabel(key: string): string | null {
  const source = props.envSources?.[key]
  if (!source || source === 'manual') return null
  return source
}

const emit = defineEmits<{
  save: [env: Record<string, string>]
}>()

const draft = ref<EnvRow[]>([])
let nextEnvRowId = 0

function nextLocalId(): string {
  nextEnvRowId += 1
  return `env-row-${nextEnvRowId}`
}

watch(
  () => props.env,
  (env) => {
    draft.value = Object.entries(env || {}).map(([key, value], index) => ({
      localId: draft.value[index]?.localId ?? nextLocalId(),
      key,
      value,
    }))
  },
  { immediate: true, deep: true },
)

function addRow() {
  draft.value.push({ localId: nextLocalId(), key: '', value: '' })
}

function removeRow(index: number) {
  draft.value.splice(index, 1)
}

function save() {
  const env = Object.fromEntries(
    draft.value.map((row) => [row.key.trim(), row.value] as const).filter(([key]) => key),
  )
  emit('save', env)
}
</script>

<template>
  <div class="env-editor" :class="{ compact: compact }">
    <div class="env-editor-header">
      <span class="env-editor-title">{{ title }}</span>
      <button v-if="!readOnly" class="env-add" @click="addRow">+ Var</button>
    </div>

    <div v-if="draft.length === 0" class="env-empty">
      {{ readOnly ? 'No annotation env vars' : 'No env vars configured' }}
    </div>

    <div v-for="(row, index) in draft" :key="row.localId" class="env-row">
      <div class="env-key-cell">
        <input
          v-model="row.key"
          class="env-input env-key"
          type="text"
          placeholder="KEY"
          :disabled="readOnly"
        />
        <span
          v-if="sourceLabel(row.key)"
          class="env-source-badge"
          :title="`Value fetched from ${sourceLabel(row.key)}. An edit here overrides it and is saved to notebook.toml (secret-looking names blanked).`"
        >
          {{ sourceLabel(row.key) }}
        </span>
      </div>
      <input
        v-model="row.value"
        class="env-input env-value"
        :type="isSensitiveKey(row.key) ? 'password' : 'text'"
        placeholder="value"
        :disabled="readOnly"
      />
      <button v-if="!readOnly" class="env-remove" @click="removeRow(index)">×</button>
    </div>

    <div v-if="!readOnly" class="env-actions">
      <button class="env-save" @click="save">Save env</button>
    </div>
  </div>
</template>

<style scoped>
.env-editor {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.env-editor.compact {
  gap: 6px;
}

.env-editor-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.env-editor-title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-secondary);
}

.env-add,
.env-save,
.env-remove {
  background: var(--bg-input);
  border: 1px solid var(--border-strong);
  color: var(--text-primary);
  border-radius: 6px;
  cursor: pointer;
}

.env-add,
.env-save {
  padding: 6px 10px;
  font-size: 12px;
}

.env-remove {
  width: 28px;
  min-width: 28px;
  height: 30px;
  font-size: 16px;
}

.env-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.env-key-cell {
  flex: 1 1 100px;
  min-width: 80px;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.env-key-cell .env-key {
  flex: none;
}
.env-key {
  flex: 1 1 100px;
  min-width: 80px;
}
.env-source-badge {
  align-self: flex-start;
  background: var(--tint-success);
  color: var(--accent-success);
  border: 1px solid var(--tint-success);
  padding: 0 6px;
  font-size: 9px;
  font-weight: 600;
  border-radius: 3px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  cursor: help;
}

.env-value {
  flex: 2 1 140px;
  min-width: 100px;
}

.env-input {
  width: 100%;
  min-width: 0;
  padding: 6px 8px;
  background: var(--bg-base);
  border: 1px solid var(--bg-input);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 12px;
}

.env-empty {
  color: var(--text-muted);
  font-size: 12px;
}

.env-actions {
  display: flex;
  justify-content: flex-end;
}
</style>
