<script setup lang="ts">
import { computed } from 'vue'
import { useNotebook } from '../stores/notebook'
import type { Cell, WidgetDescriptor } from '../types/notebook'

/**
 * Renders a widget cell's control panel. Each control is declared in the cell
 * source (`name = slider(...)`) and its value lives in runtime state. Changing
 * a control sends a debounced `widget_update` over the WS, which re-materializes
 * the value artifact and stales downstream cells.
 */
const props = defineProps<{ cell: Cell }>()
const { updateWidgetValues, updateSource, flushCellSource } = useNotebook()

const descriptors = computed<WidgetDescriptor[]>(() => props.cell.widget?.descriptors ?? [])

// Live mode: whether the cell source carries a `# @live` annotation. When on,
// changing a control auto-runs the cheap downstream cells (backend cost-gated).
const isLive = computed(() =>
  props.cell.source.split('\n').some((line) => {
    const match = /^#\s*@live\b\s*(\w+)?/.exec(line.trim())
    return Boolean(match) && !['off', 'false', 'no', '0'].includes((match?.[1] ?? '').toLowerCase())
  }),
)

function toggleLive() {
  const lines = props.cell.source.split('\n')
  const idx = lines.findIndex((line) => /^#\s*@live\b/.test(line.trim()))
  const next =
    idx >= 0 ? lines.filter((_, i) => i !== idx).join('\n') : `# @live\n${props.cell.source}`
  updateSource(props.cell.id, next)
  flushCellSource(props.cell.id)
}

function currentValue(d: WidgetDescriptor): unknown {
  const v = props.cell.widget?.values?.[d.name]
  return v === undefined ? d.default : v
}
function num(value: unknown, fallback = 0): number {
  return typeof value === 'number' ? value : fallback
}

// Debounce control changes so a slider drag is a train of frames flushed on
// release; discrete controls (dropdown/checkbox/number) push immediately.
let timer: ReturnType<typeof setTimeout> | undefined
function push(name: string, value: unknown, immediate = false) {
  clearTimeout(timer)
  const send = () => updateWidgetValues(props.cell.id, { [name]: value })
  if (immediate) send()
  else timer = setTimeout(send, 250)
}

function onSliderInput(d: WidgetDescriptor, e: Event) {
  push(d.name, Number((e.target as HTMLInputElement).value))
}
function onSliderCommit(d: WidgetDescriptor, e: Event) {
  push(d.name, Number((e.target as HTMLInputElement).value), true)
}
function onNumber(d: WidgetDescriptor, e: Event) {
  push(d.name, Number((e.target as HTMLInputElement).value), true)
}
function onSelect(d: WidgetDescriptor, e: Event) {
  push(d.name, (e.target as HTMLSelectElement).value, true)
}
function onText(d: WidgetDescriptor, e: Event) {
  push(d.name, (e.target as HTMLInputElement).value)
}
function onCheckbox(d: WidgetDescriptor, e: Event) {
  push(d.name, (e.target as HTMLInputElement).checked, true)
}
</script>

<template>
  <div class="widget-cell">
    <div class="widget-toolbar">
      <button
        type="button"
        class="live-toggle"
        :class="{ on: isLive }"
        :title="
          isLive
            ? 'Live: control changes auto-run downstream cells'
            : 'Off: control changes mark downstream stale (run manually)'
        "
        @click="toggleLive"
      >
        ⚡ Live{{ isLive ? ' on' : ' off' }}
      </button>
    </div>
    <div v-if="!descriptors.length" class="widget-empty">
      No controls. Declare one per line, e.g. <code>alpha = slider(0, 1, default=0.5)</code>.
    </div>
    <div v-for="d in descriptors" :key="d.name" class="widget-control">
      <label class="widget-label">{{ d.name }}</label>

      <template v-if="d.kind === 'slider'">
        <input
          type="range"
          :min="num(d.params.min)"
          :max="num(d.params.max, 1)"
          :step="num(d.params.step, 0.01) || 'any'"
          :value="num(currentValue(d))"
          @input="onSliderInput(d, $event)"
          @change="onSliderCommit(d, $event)"
        />
        <span class="widget-value">{{ currentValue(d) }}</span>
      </template>

      <template v-else-if="d.kind === 'number'">
        <input
          type="number"
          :min="d.params.min as number | undefined"
          :max="d.params.max as number | undefined"
          :value="num(currentValue(d))"
          @change="onNumber(d, $event)"
        />
      </template>

      <template v-else-if="d.kind === 'dropdown'">
        <select :value="String(currentValue(d))" @change="onSelect(d, $event)">
          <option
            v-for="opt in (d.params.options as unknown[]) || []"
            :key="String(opt)"
            :value="String(opt)"
          >
            {{ opt }}
          </option>
        </select>
      </template>

      <template v-else-if="d.kind === 'checkbox'">
        <input
          type="checkbox"
          :checked="Boolean(currentValue(d))"
          @change="onCheckbox(d, $event)"
        />
      </template>

      <template v-else-if="d.kind === 'text'">
        <input type="text" :value="String(currentValue(d) ?? '')" @input="onText(d, $event)" />
      </template>
    </div>
  </div>
</template>

<style scoped>
.widget-cell {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
}
.widget-toolbar {
  display: flex;
  justify-content: flex-end;
}
.live-toggle {
  background: var(--bg-input);
  color: var(--text-muted);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 12px;
  cursor: pointer;
}
.live-toggle.on {
  color: var(--accent-primary);
  border-color: var(--accent-primary);
}
.widget-empty {
  color: var(--text-muted);
  font-size: 13px;
}
.widget-empty code {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  background: var(--bg-input);
  padding: 1px 5px;
  border-radius: 4px;
}
.widget-control {
  display: grid;
  grid-template-columns: 140px 1fr auto;
  align-items: center;
  gap: 12px;
}
.widget-label {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: var(--accent-primary);
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
}
.widget-control input[type='range'] {
  width: 100%;
  accent-color: var(--accent-primary);
}
.widget-control input[type='number'],
.widget-control input[type='text'],
.widget-control select {
  background: var(--bg-input);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 3px 8px;
  font: inherit;
}
.widget-value {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: var(--text-primary);
  min-width: 48px;
  text-align: right;
}
</style>
