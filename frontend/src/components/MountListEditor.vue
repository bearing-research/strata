<script setup lang="ts">
import { X } from '@lucide/vue'
import { ref, watch } from 'vue'
import type { MountSpec } from '../types/notebook'

const props = withDefaults(
  defineProps<{
    mounts: MountSpec[]
    title?: string
    compact?: boolean
    readOnly?: boolean
    showPin?: boolean
  }>(),
  {
    title: 'Mounts',
    compact: false,
    readOnly: false,
    showPin: true,
  },
)

const emit = defineEmits<{
  save: [mounts: MountSpec[]]
}>()

type DraftMount = MountSpec & { _key: string }

const draft = ref<DraftMount[]>([])
let nextDraftMountKey = 0

function newDraftMount(mount?: MountSpec): DraftMount {
  return {
    name: mount?.name ?? '',
    uri: mount?.uri ?? '',
    mode: mount?.mode ?? 'ro',
    pin: mount?.pin ?? null,
    _key: `mount-${nextDraftMountKey++}`,
  }
}

function cloneMounts(mounts: MountSpec[]): DraftMount[] {
  return mounts.map((mount) => newDraftMount(mount))
}

watch(
  () => props.mounts,
  (mounts) => {
    draft.value = cloneMounts(mounts)
  },
  { immediate: true, deep: true },
)

function addMount() {
  draft.value.push(newDraftMount())
}

function removeMount(index: number) {
  draft.value.splice(index, 1)
}

function save() {
  emit(
    'save',
    draft.value
      .map((mount) => ({
        name: mount.name.trim(),
        uri: mount.uri.trim(),
        mode: mount.mode,
        pin: mount.pin?.trim() || null,
      }))
      .filter((mount) => mount.name && mount.uri),
  )
}
</script>

<template>
  <div class="mount-editor" :class="{ compact: compact }">
    <div class="mount-editor-header">
      <span class="mount-editor-title">{{ title }}</span>
      <button v-if="!readOnly" class="mount-add" @click="addMount">+ Mount</button>
    </div>

    <div v-if="draft.length === 0" class="mount-empty">
      {{ readOnly ? 'No source annotations' : 'No mounts configured' }}
    </div>

    <div
      v-for="(mount, index) in draft"
      :key="mount._key"
      class="mount-row"
      :class="{ 'mount-row-no-pin': !showPin }"
    >
      <input
        v-model="mount.name"
        class="mount-input mount-name"
        type="text"
        placeholder="name"
        :disabled="readOnly"
      />
      <input
        v-model="mount.uri"
        class="mount-input mount-uri"
        type="text"
        placeholder="file:///... or s3://..."
        :disabled="readOnly"
      />
      <select v-model="mount.mode" class="mount-input mount-mode" :disabled="readOnly">
        <option value="ro">ro</option>
        <option value="rw">rw</option>
      </select>
      <input
        v-if="showPin"
        v-model="mount.pin"
        class="mount-input mount-pin"
        type="text"
        placeholder="pin (optional)"
        :disabled="readOnly"
      />
      <button
        v-if="!readOnly"
        class="mount-remove"
        aria-label="Remove mount"
        @click="removeMount(index)"
      >
        <X />
      </button>
    </div>

    <div v-if="!readOnly" class="mount-actions">
      <button class="mount-save" @click="save">Save mounts</button>
    </div>
  </div>
</template>

<style scoped>
.mount-editor {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.mount-editor.compact {
  gap: 6px;
}

.mount-editor-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.mount-editor-title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-secondary);
}

.mount-add,
.mount-save,
.mount-remove {
  background: var(--bg-input);
  border: 1px solid var(--border-strong);
  color: var(--text-primary);
  border-radius: 6px;
  cursor: pointer;
}

.mount-add,
.mount-save {
  padding: 6px 10px;
  font-size: 12px;
}

.mount-remove {
  width: 28px;
  min-width: 28px;
  height: 30px;
  font-size: 16px;
}

.mount-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.mount-name {
  flex: 1 1 120px;
}

.mount-uri {
  flex: 2 1 260px;
}

.mount-mode {
  flex: 0 0 72px;
}

.mount-pin {
  flex: 1 1 140px;
}

.mount-remove {
  flex: 0 0 28px;
}

.compact .mount-name {
  flex-basis: 96px;
}

.compact .mount-uri {
  flex-basis: 200px;
}

.compact .mount-mode {
  flex-basis: 64px;
}

.compact .mount-pin {
  flex-basis: 120px;
}

.mount-input {
  width: 100%;
  min-width: 0;
  padding: 6px 8px;
  background: var(--bg-input);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 12px;
}

.mount-input:disabled {
  opacity: 0.85;
}

.mount-empty {
  color: var(--text-muted);
  font-size: 12px;
}

.mount-actions {
  display: flex;
  justify-content: flex-end;
}

@media (max-width: 920px) {
  .mount-mode,
  .compact .mount-mode {
    flex-basis: 96px;
  }

  .mount-remove {
    margin-left: auto;
  }
}
</style>
