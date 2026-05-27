<script setup lang="ts">
import { computed, defineAsyncComponent, onUnmounted, ref, watch } from 'vue'
import { useCodemirror } from '../composables/useCodemirror'
import { useNotebook } from '../stores/notebook'
import type { Cell, CellOutput } from '../types/notebook'
import {
  resolveEffectiveWorkerEntry,
  summarizeRemoteExecutionState,
  summarizeRemoteExecutionIssue,
  workerTransportLabel,
  workerWarningForEntry,
} from '../utils/notebookWorkers'
import { renderMarkdownToHtml } from '../utils/markdown'

const InspectPanel = defineAsyncComponent(() => import('./InspectPanel.vue'))

const props = defineProps<{ cell: Cell }>()
const emit = defineEmits<{
  run: [cellId: string]
  rerun: [cellId: string]
  delete: [cellId: string]
  addBelow: [cellId: string]
  duplicate: [cellId: string]
  moveUp: [cellId: string]
  moveDown: [cellId: string]
}>()

const {
  notebook,
  connected,
  environmentMutationActive,
  environmentLastAction,
  environmentOperation,
  updateSource,
  flushCellSource,
  openInspect,
  isInspecting: storeIsInspecting,
  closeInspect,
  availableWorkers,
  addDependencyAction,
  setVariantActive,
  addVariant,
  cancelCellWebSocket,
} = useNotebook()

const variantGroup = computed(() => {
  if (!props.cell.variantGroup) return null
  return notebook.variantGroups.find((g) => g.group === props.cell.variantGroup) ?? null
})

const isInspecting = computed(() => storeIsInspecting(props.cell.id))

function toggleInspect() {
  if (isInspecting.value) {
    closeInspect(props.cell.id)
    return
  }
  openInspect(props.cell.id)
}

const editorEl = ref<HTMLElement | null>(null)
const showCausality = ref(false)
const folded = ref(false)
const installRequestedPackage = ref<string | null>(null)

// Markdown cells default to a rendered "preview" view; clicking the
// preview swaps in the CodeMirror editor for editing. New (empty)
// markdown cells start in edit mode so the user can begin typing
// immediately. Non-markdown cells ignore this state entirely.
const isMarkdownPreviewing = ref(
  props.cell.language === 'markdown' && Boolean(props.cell.source.trim()),
)

const renderedMarkdownSource = computed(() => {
  if (props.cell.language !== 'markdown' || !props.cell.source.trim()) {
    return ''
  }
  return renderMarkdownToHtml(props.cell.source)
})

function enterMarkdownEdit() {
  if (props.cell.language !== 'markdown') return
  isMarkdownPreviewing.value = false
  // Focus the editor on the next tick so the click handler doesn't race
  // with CodeMirror's mount/show transition.
  setTimeout(() => view.value?.focus(), 0)
}

function exitMarkdownEditOnBlur() {
  // Persist any pending source first; blur is when the debounced flush
  // would otherwise lose un-WS'd edits.
  flushCellSource(props.cell.id)
  if (props.cell.language === 'markdown' && props.cell.source.trim()) {
    isMarkdownPreviewing.value = true
  }
}

const { view, setDoc } = useCodemirror(editorEl, {
  initialDoc: props.cell.source,
  language: props.cell.language,
  onUpdate: (doc) => updateSource(props.cell.id, doc),
  onRun: () => emit('run', props.cell.id),
  onRerun: () => emit('rerun', props.cell.id),
})

watch(
  () => props.cell.source,
  (source) => {
    if (view.value && view.value.state.doc.toString() !== source) {
      setDoc(source)
    }
  },
)

watch(
  () => props.cell.status,
  (status) => {
    if (status === 'running') {
      installRequestedPackage.value = null
    }
  },
)

const statusClass = computed(() => `status-${props.cell.status}`)

// Live dispatch badge: shows "dispatching → gpu-fly" when a remote cell
// is in-flight. The backend includes remote_worker on the cell_status
// running message precisely so this badge can appear without waiting for
// the cell to finish.
const dispatchLabel = computed(() => {
  if (props.cell.status !== 'running') return null
  const worker = props.cell.remoteWorkerName
  if (!worker) return null
  return `dispatching → ${worker}`
})

const statusLabel = computed(() => {
  switch (props.cell.status) {
    case 'idle':
      return '\u25CB'
    case 'queued':
      return '\u25F7'
    case 'running':
      return '\u25CC'
    case 'ready':
      return '\u25CF'
    case 'stale':
      return '\u25D0'
    case 'error':
      return '\u2715'
  }
})

const durationLabel = computed(() => {
  if (!props.cell.durationMs) return ''
  const ms = props.cell.durationMs
  const time = ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`
  if (props.cell.output?.cacheHit) return time
  return `${time}`
})

const canExplainStaleness = computed(() =>
  Boolean(props.cell.causality && (props.cell.status === 'idle' || props.cell.status === 'stale')),
)

/** v1.1: Execution method label */
const executionMethodLabel = computed(() => {
  if (!props.cell.executorName) return ''
  switch (props.cell.executorName) {
    case 'cached':
      return 'cached'
    case 'warm':
      return 'warm'
    case 'cold':
      return 'cold'
    default:
      return props.cell.executorName
  }
})

/**
 * Phase 2: Progress badge for loop cells.
 *
 * While the loop is running, ``# @loop_until`` has not yet fired, and the
 * current iteration is strictly less than ``maxIter``, we display the
 * completed-iteration count with a spinner so the user sees live progress
 * of a multi-minute agentic loop. After the loop finishes the badge stays
 * visible as a compact summary ("iter N/M done" or similar) so users can
 * see at a glance that a cell ran 7 iterations without opening the
 * inspect panel.
 */
const loopProgressDone = computed(() => {
  const progress = props.cell.loopProgress
  if (!progress) return false
  if (progress.untilReached) return true
  return progress.iteration >= progress.maxIter - 1
})

const loopProgressLabel = computed(() => {
  const progress = props.cell.loopProgress
  if (!progress) return ''
  // iteration is 0-based; display the 1-based completed-iteration count.
  const completed = progress.iteration + 1
  return `iter ${completed}/${progress.maxIter}`
})

const loopProgressTitle = computed(() => {
  const progress = props.cell.loopProgress
  if (!progress) return ''
  const completed = progress.iteration + 1
  const status = progress.untilReached
    ? 'loop_until fired — loop complete'
    : completed >= progress.maxIter
      ? 'reached max_iter — loop complete'
      : props.cell.status === 'running'
        ? 'loop running'
        : 'loop paused'
  const duration =
    typeof progress.iterDurationMs === 'number'
      ? ` · last iter ${Math.round(progress.iterDurationMs)}ms`
      : ''
  return `${status} (iter ${completed}/${progress.maxIter})${duration}`
})

// ---- Elapsed-time tracking for running cells ----
//
// Plain Python cells get no per-step progress signal (only loop cells
// emit cell_iteration_progress). Without an elapsed counter, a cell
// that's been training for 20 minutes looks identical to one that
// started two seconds ago: same status dot, no time. Track the
// transition into 'running' and tick a per-cell counter while it's
// in that state.
const runStartedAtMs = ref<number | null>(null)
const nowMs = ref(Date.now())
let tickHandle: number | null = null

function startTicker() {
  if (tickHandle !== null) return
  nowMs.value = Date.now()
  tickHandle = window.setInterval(() => {
    nowMs.value = Date.now()
  }, 1000)
}

function stopTicker() {
  if (tickHandle !== null) {
    clearInterval(tickHandle)
    tickHandle = null
  }
}

onUnmounted(stopTicker)

watch(
  () => props.cell.status,
  (status, prev) => {
    if (status === 'running' && prev !== 'running') {
      // Reload edge case: status was already 'running' when the
      // component mounted. We don't know the real start time, so this
      // counts from "first observed running", a lower bound. Better
      // than no info.
      runStartedAtMs.value = Date.now()
      startTicker()
    } else if (status !== 'running') {
      runStartedAtMs.value = null
      stopTicker()
    }
  },
  { immediate: true },
)

function _formatElapsed(ms: number): string {
  const secs = Math.max(0, Math.floor(ms / 1000))
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  const remSecs = secs % 60
  if (mins < 60) return `${mins}m ${remSecs}s`
  const hrs = Math.floor(mins / 60)
  const remMins = mins % 60
  return `${hrs}h ${remMins}m`
}

const runningElapsedLabel = computed(() => {
  if (props.cell.status !== 'running' || runStartedAtMs.value === null) return ''
  return _formatElapsed(nowMs.value - runStartedAtMs.value)
})

const runningElapsedTitle = computed(() => {
  if (!runningElapsedLabel.value) return ''
  return `Cell has been running for ${runningElapsedLabel.value}. Click the stop button in the gutter to cancel.`
})

function cancelThisCell() {
  cancelCellWebSocket(props.cell.id)
}

/** v1.1: Causality summary for tooltip */
const causalityTooltip = computed(() => {
  const c = props.cell.causality
  if (!c) return ''
  return c.details
    .map((d) => {
      switch (d.type) {
        case 'source_changed':
          return `source of "${d.cellName || d.cellId}" changed`
        case 'input_changed':
          return `upstream "${d.cellName || d.cellId}" changed`
        case 'env_changed':
          return `environment changed (${d.package || 'uv.lock'})`
        default:
          return d.type
      }
    })
    .join('; ')
})

function toggleCausality() {
  if (!canExplainStaleness.value) {
    showCausality.value = false
    return
  }
  showCausality.value = !showCausality.value
}

watch(canExplainStaleness, (canExplain) => {
  if (!canExplain) {
    showCausality.value = false
  }
})

const mountSummary = computed(() =>
  props.cell.mounts.map((mount) => `${mount.name}:${mount.mode}`).join(', '),
)

const effectiveWorkerLabel = computed(
  () => props.cell.annotations?.worker || props.cell.worker || 'local',
)
const effectiveWorkerEntry = computed(() =>
  resolveEffectiveWorkerEntry(availableWorkers.value, effectiveWorkerLabel.value),
)
const effectiveWorkerHealthLabel = computed(() => {
  const entry = effectiveWorkerEntry.value
  if (!entry || entry.backend === 'local') return null
  return entry.health
})
const effectiveWorkerTransportLabel = computed(() => {
  const entry = effectiveWorkerEntry.value
  if (!entry || entry.backend === 'local') return null
  return workerTransportLabel(entry)
})
const lastRemoteWorkerLabel = computed(() => props.cell.remoteWorkerName || null)
const lastRemoteTransportLabel = computed(() => props.cell.remoteTransport || null)
const lastRemoteBuildId = computed(() => props.cell.remoteBuildId || null)
const lastRemoteBuildState = computed(() => props.cell.remoteBuildState || null)
const lastRemoteErrorCode = computed(() => props.cell.remoteErrorCode || null)
const workerWarning = computed(() => {
  return workerWarningForEntry(effectiveWorkerEntry.value, effectiveWorkerLabel.value)
})
const remoteExecutionIssueSummary = computed(() => {
  return summarizeRemoteExecutionIssue(
    props.cell.output?.error || '',
    effectiveWorkerEntry.value,
    effectiveWorkerLabel.value,
    lastRemoteErrorCode.value,
    lastRemoteBuildState.value,
  )
})
const remoteExecutionSummary = computed(() =>
  summarizeRemoteExecutionState({
    executionMethod: props.cell.executorName,
    remoteWorkerName: lastRemoteWorkerLabel.value,
    remoteTransport: lastRemoteTransportLabel.value,
    remoteBuildState: lastRemoteBuildState.value,
    remoteErrorCode: lastRemoteErrorCode.value,
    hasError: Boolean(props.cell.output?.error),
  }),
)
const showRemoteExecutionSummary = computed(() =>
  Boolean(
    remoteExecutionSummary.value &&
    (lastRemoteWorkerLabel.value ||
      lastRemoteTransportLabel.value ||
      lastRemoteBuildId.value ||
      lastRemoteBuildState.value ||
      lastRemoteErrorCode.value ||
      props.cell.executorName === 'executor' ||
      props.cell.output?.cacheHit),
  ),
)
const effectiveTimeoutLabel = computed(() => {
  const timeout = props.cell.annotations?.timeout ?? props.cell.timeout
  return timeout == null ? null : `${timeout}s`
})
const effectiveEnvCount = computed(
  () =>
    Object.keys({
      ...props.cell.env,
      ...(props.cell.annotations?.env || {}),
    }).length,
)

const hasAnnotationOverrides = computed(() => {
  const annotations = props.cell.annotations
  return Boolean(
    annotations &&
    (annotations.mounts.length ||
      annotations.worker ||
      annotations.timeout != null ||
      Object.keys(annotations.env).length),
  )
})

const annotationSummaryChips = computed(() => {
  const annotations = props.cell.annotations
  if (!annotations) return []

  const chips: string[] = []
  if (annotations.worker) chips.push(`@worker ${annotations.worker}`)
  if (annotations.timeout != null) chips.push(`@timeout ${annotations.timeout}s`)
  const envCount = Object.keys(annotations.env).length
  if (envCount) chips.push(`@env ${envCount}`)
  if (annotations.mounts.length) chips.push(`@mount ${annotations.mounts.length}`)
  return chips
})

const visibleAnnotationSummaryChips = computed(() => annotationSummaryChips.value.slice(0, 2))
const hiddenAnnotationSummaryCount = computed(() =>
  Math.max(0, annotationSummaryChips.value.length - visibleAnnotationSummaryChips.value.length),
)
const annotationSummaryTitle = computed(() => {
  if (!annotationSummaryChips.value.length) {
    return 'Source annotations override saved config'
  }
  return `Source annotations: ${annotationSummaryChips.value.join(' · ')}`
})

const annotationDiagnostics = computed(() => props.cell.annotationDiagnostics ?? [])
const annotationDiagnosticsLabel = computed(() => {
  const n = annotationDiagnostics.value.length
  if (!n) return ''
  return n === 1 ? '1 annotation issue' : `${n} annotation issues`
})
const annotationDiagnosticsTitle = computed(() =>
  annotationDiagnostics.value.map((d) => `${d.code}: ${d.message}`).join('\n'),
)

// Module-cell marker. Only shown when the cell classifies as a module
// cell (pure source + at least one exported def/class). The tooltip
// lists the symbols the cell makes available to downstream cells —
// the same names they'd reference by bare identifier.
const moduleExportsTitle = computed(() => {
  const exports = props.cell.moduleExports
  if (!exports?.length) {
    return 'Module cell — definitions here can be referenced from downstream cells.'
  }
  const lines = exports.map((e) => `  ${e.kind} ${e.name}`).join('\n')
  return `Module cell — downstream cells can reference:\n${lines}`
})

function normalizePackageName(pkg: string | null | undefined): string {
  return (pkg || '').trim().toLowerCase()
}

const installTargetPackage = computed(
  () => installRequestedPackage.value || props.cell.suggestInstall || null,
)

const installInProgress = computed(() => {
  return (
    environmentMutationActive.value &&
    environmentOperation.value?.action === 'add' &&
    normalizePackageName(environmentOperation.value.packageName) ===
      normalizePackageName(installTargetPackage.value)
  )
})

const installCompleted = computed(() => {
  return (
    !!installRequestedPackage.value &&
    !props.cell.suggestInstall &&
    environmentLastAction.value?.action === 'add' &&
    normalizePackageName(environmentLastAction.value.packageName) ===
      normalizePackageName(installRequestedPackage.value)
  )
})

async function installSuggestedPackage() {
  const pkg = props.cell.suggestInstall
  if (!pkg || installInProgress.value) return
  installRequestedPackage.value = pkg
  await addDependencyAction(pkg)
}

const outputExpanded = ref(false)
const COMPACT_LINE_LIMIT = 10

function isLongText(text: string | null | undefined): boolean {
  if (!text) return false
  return text.split('\n').length > COMPACT_LINE_LIMIT || text.length > 800
}

function compactText(text: string): string {
  const lines = text.split('\n')
  if (lines.length <= COMPACT_LINE_LIMIT && text.length <= 800) return text
  return lines.slice(0, COMPACT_LINE_LIMIT).join('\n')
}

function overflowSummary(text: string): string {
  const totalLines = text.split('\n').length
  const hidden = totalLines - COMPACT_LINE_LIMIT
  return hidden > 0 ? `${hidden} more lines` : 'more'
}

/** Lightweight JSON syntax highlighting via regex → colored spans. */
function highlightJson(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/("(?:\\.|[^"\\])*")\s*:/g, '<span class="json-key">$1</span>:')
    .replace(/:\s*("(?:\\.|[^"\\])*")/g, ': <span class="json-string">$1</span>')
    .replace(/:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)/g, ': <span class="json-number">$1</span>')
    .replace(/:\s*(true|false|null)\b/g, ': <span class="json-bool">$1</span>')
}

function isJsonLike(text: string): boolean {
  const trimmed = text.trimStart()
  return trimmed.startsWith('{') || trimmed.startsWith('[')
}

/** Combined console text for a cell — stdout first, stderr after. Empty
 * string when the cell has no captured output. Console is a Cell-level
 * field (not part of the display output) so @output_schema cells keep
 * a clean structured display value. */
function cellConsoleText(cell: Cell): string {
  const stdout = cell.consoleStdout ?? ''
  const stderr = cell.consoleStderr ?? ''
  if (stdout && stderr) return `${stdout}\n${stderr}`
  return stdout || stderr
}

/** Format scalar output for display */
function formatScalar(scalar: unknown): string {
  if (scalar === null || scalar === undefined) return 'None'
  if (typeof scalar === 'object') {
    return JSON.stringify(scalar, null, 2)
  }
  return String(scalar)
}

const visibleOutputs = computed(() => {
  if (props.cell.displayOutputs?.length) {
    return props.cell.displayOutputs
  }
  return props.cell.output ? [props.cell.output] : []
})

function renderedMarkdownOutput(output: CellOutput): string {
  if (output.contentType !== 'text/markdown' || !output.markdownText) {
    return ''
  }
  return renderMarkdownToHtml(output.markdownText)
}

function outputKey(output: CellOutput, index: number): string {
  return `${index}:${output.artifactUri || output.contentType}`
}
</script>

<template>
  <div
    class="cell"
    :class="[
      statusClass,
      {
        'cell-prompt': cell.language === 'prompt',
        'cell-markdown': cell.language === 'markdown',
      },
    ]"
    data-testid="notebook-cell"
    :data-cell-id="cell.id"
  >
    <!-- Left gutter -->
    <div class="cell-gutter">
      <span class="status-dot" :title="cell.status">{{ statusLabel }}</span>
      <div class="cell-actions">
        <button
          v-if="cell.language !== 'markdown' && cell.status !== 'running'"
          title="Run (Shift+Enter)"
          :disabled="!connected || environmentMutationActive"
          @click="emit('run', cell.id)"
        >
          &#x25B6;
        </button>
        <button
          v-if="cell.language !== 'markdown' && cell.status !== 'running'"
          title="Rerun — force re-execute, bypass cache (Cmd+Shift+Enter)"
          data-testid="cell-rerun-button"
          :disabled="!connected || environmentMutationActive"
          @click="emit('rerun', cell.id)"
        >
          &#x21BB;
        </button>
        <button
          v-else-if="cell.language !== 'markdown'"
          class="cancel-btn"
          title="Cancel running cell"
          data-testid="cancel-cell-button"
          @click="cancelThisCell"
        >
          &#x25A0;
        </button>
        <button title="Move up" @click="emit('moveUp', cell.id)">&#x25B2;</button>
        <button title="Move down" @click="emit('moveDown', cell.id)">&#x25BC;</button>
        <button title="Add cell below" @click="emit('addBelow', cell.id)">+</button>
        <button title="Duplicate cell" @click="emit('duplicate', cell.id)">&#x2398;</button>
        <button title="Delete cell" @click="emit('delete', cell.id)">&times;</button>
        <button
          :title="folded ? 'Expand cell' : 'Collapse cell'"
          :class="{ active: folded }"
          @click="folded = !folded"
        >
          {{ folded ? '&#x25B7;' : '&#x25BD;' }}
        </button>
        <button title="Inspect inputs" :class="{ active: isInspecting }" @click="toggleInspect">
          &#x1F50D;
        </button>
      </div>
    </div>

    <!-- Editor + output -->
    <div class="cell-body">
      <!-- Variant tab strip — shown when this cell is part of a variant group -->
      <div v-if="variantGroup" class="variant-tabs" data-testid="variant-tabs">
        <button
          v-for="member in variantGroup.members"
          :key="member.cellId"
          class="variant-tab"
          :class="{ active: member.cellId === variantGroup.activeCellId }"
          :title="
            member.cellId === variantGroup.activeCellId
              ? `Active variant: ${member.name}`
              : `Switch to ${member.name}`
          "
          :data-variant-name="member.name"
          @click="setVariantActive(variantGroup.group, member.name)"
        >
          {{ member.name }}
        </button>
        <button
          class="variant-tab variant-tab-add"
          title="Add a new variant (clones the active one)"
          data-testid="variant-tab-add"
          @click="addVariant(variantGroup.group)"
        >
          +
        </button>
        <span class="variant-group-label" :title="`Variant group: ${variantGroup.group}`">
          {{ variantGroup.group }}
        </span>
      </div>
      <div class="cell-meta">
        <!-- Line 1: identity — name, defines, reads -->
        <div class="cell-meta-row">
          <div class="cell-meta-main">
            <span class="cell-lang">{{ cell.language }}</span>
            <span
              v-if="cell.annotations?.name"
              class="name-badge"
              :title="`Cell name: ${cell.annotations.name}`"
            >
              {{ cell.annotations.name }}
            </span>
            <span
              v-if="cell.isLeaf"
              class="leaf-badge"
              title="This cell is a leaf (no downstream consumers)"
              >leaf</span
            >
            <span v-if="cell.defines.length" class="cell-vars">
              defines: <code>{{ cell.defines.join(', ') }}</code>
            </span>
            <span
              v-if="cell.shadowWarnings && cell.shadowWarnings.length"
              class="shadow-badge"
              :title="cell.shadowWarnings.join('\n')"
            >
              shadows
            </span>
            <span v-if="cell.upstreamIds.length" class="cell-vars">
              reads: <code>{{ cell.references.join(', ') }}</code>
            </span>
          </div>
          <div class="cell-meta-actions">
            <span v-if="durationLabel" class="duration">{{ durationLabel }}</span>
            <button
              v-if="canExplainStaleness"
              class="causality-btn"
              :title="causalityTooltip"
              @click="toggleCausality"
            >
              Why stale?
            </button>
          </div>
        </div>
        <!-- Line 2: runtime — worker, mounts, timeout, env, annotations, cache -->
        <div class="cell-meta-row cell-meta-runtime">
          <span class="worker-badge" :title="`Worker: ${effectiveWorkerLabel}`">
            worker: {{ effectiveWorkerLabel }}
          </span>
          <span
            v-if="dispatchLabel"
            class="dispatch-badge"
            :title="'Cell is executing on a remote worker'"
          >
            {{ dispatchLabel }}
          </span>
          <span
            v-if="effectiveWorkerTransportLabel"
            class="worker-transport-badge"
            :title="`Remote worker transport: ${effectiveWorkerTransportLabel}`"
          >
            {{ effectiveWorkerTransportLabel }}
          </span>
          <span
            v-if="effectiveWorkerHealthLabel"
            class="worker-health-badge"
            :class="{ warning: workerWarning }"
            :title="workerWarning || `Worker health: ${effectiveWorkerHealthLabel}`"
          >
            {{ workerWarning ? 'attention' : effectiveWorkerHealthLabel }}
          </span>
          <span v-if="cell.mounts.length" class="mount-badge" :title="mountSummary">
            mounts: {{ cell.mounts.length }}
          </span>
          <span
            v-if="effectiveTimeoutLabel"
            class="timeout-badge"
            :title="`Timeout: ${effectiveTimeoutLabel}`"
          >
            timeout: {{ effectiveTimeoutLabel }}
          </span>
          <span v-if="effectiveEnvCount" class="env-badge" :title="`${effectiveEnvCount} env vars`">
            env: {{ effectiveEnvCount }}
          </span>
          <span
            v-if="hasAnnotationOverrides"
            class="annotation-badge"
            :title="annotationSummaryTitle"
          >
            source overrides
          </span>
          <span
            v-for="chip in visibleAnnotationSummaryChips"
            :key="chip"
            class="annotation-summary-chip"
            :title="annotationSummaryTitle"
          >
            {{ chip }}
          </span>
          <span
            v-if="hiddenAnnotationSummaryCount"
            class="annotation-summary-chip"
            :title="annotationSummaryTitle"
          >
            +{{ hiddenAnnotationSummaryCount }}
          </span>
          <span
            v-if="annotationDiagnosticsLabel"
            class="annotation-diagnostic-badge"
            :title="annotationDiagnosticsTitle"
          >
            &#x26A0; {{ annotationDiagnosticsLabel }}
          </span>
          <span v-if="cell.isModuleCell" class="module-cell-badge" :title="moduleExportsTitle">
            module
          </span>
          <span
            v-if="loopProgressLabel"
            class="loop-progress-badge"
            :class="{
              running: cell.status === 'running',
              done: loopProgressDone,
            }"
            :title="loopProgressTitle"
          >
            <span
              v-if="cell.status === 'running' && !loopProgressDone"
              class="loop-spinner"
              aria-hidden="true"
            ></span>
            &#x21BB; {{ loopProgressLabel }}
          </span>
          <span
            v-if="cell.status === 'running' && !loopProgressLabel"
            class="running-badge"
            data-testid="running-elapsed"
            :title="runningElapsedTitle"
          >
            <span class="running-spinner" aria-hidden="true"></span>
            running · {{ runningElapsedLabel }}
          </span>
          <span
            v-if="cell.output?.cacheHit"
            class="cache-badge"
            :title="
              cell.output?.cacheLoadMs
                ? `Result loaded from cache in ${cell.output.cacheLoadMs}ms`
                : 'Result loaded from cache'
            "
          >
            &#x26A1; cached{{ cell.output?.cacheLoadMs ? ` · ${cell.output.cacheLoadMs}ms` : '' }}
          </span>
          <span
            v-if="executionMethodLabel && !cell.output?.cacheHit"
            class="exec-method-badge"
            :title="`Executed via ${executionMethodLabel} process`"
          >
            {{ executionMethodLabel }}
          </span>
        </div>
      </div>

      <div
        v-if="!folded && showRemoteExecutionSummary && remoteExecutionSummary"
        class="remote-execution-summary"
        :class="`tone-${remoteExecutionSummary.tone}`"
      >
        <span class="remote-execution-label">{{ remoteExecutionSummary.label }}</span>
        <span class="remote-execution-detail">{{ remoteExecutionSummary.detail }}</span>
        <span v-if="lastRemoteWorkerLabel" class="remote-execution-pill">
          {{ lastRemoteWorkerLabel }}
        </span>
        <span v-if="lastRemoteTransportLabel" class="remote-execution-pill">
          {{ lastRemoteTransportLabel }}
        </span>
        <span v-if="lastRemoteBuildId" class="remote-execution-pill">
          build {{ lastRemoteBuildId }}
        </span>
        <span v-if="lastRemoteBuildState" class="remote-execution-pill">
          {{ lastRemoteBuildState }}
        </span>
        <span v-if="lastRemoteErrorCode" class="remote-execution-pill">
          {{ lastRemoteErrorCode }}
        </span>
        <span v-if="effectiveWorkerHealthLabel" class="remote-execution-pill">
          {{ effectiveWorkerHealthLabel }}
        </span>
      </div>

      <!-- v1.1: Causality chain detail (expanded) -->
      <div v-if="showCausality && canExplainStaleness && cell.causality" class="causality-panel">
        <div class="causality-header">
          Stale because: <span class="causality-reason">{{ cell.causality.reason }}</span>
        </div>
        <ul class="causality-details">
          <li v-for="(detail, i) in cell.causality.details" :key="i" class="causality-detail">
            <span v-if="detail.type === 'source_changed'" class="detail-icon">&#x270E;</span>
            <span v-else-if="detail.type === 'input_changed'" class="detail-icon">&#x2191;</span>
            <span v-else class="detail-icon">&#x2699;</span>
            <span v-if="detail.type === 'source_changed'">
              Source of <code>{{ detail.cellName || detail.cellId }}</code> changed
            </span>
            <span v-else-if="detail.type === 'input_changed'">
              Upstream <code>{{ detail.cellName || detail.cellId }}</code> changed
              <span v-if="detail.fromVersion" class="version-change">
                {{ detail.fromVersion }} &rarr; {{ detail.toVersion }}
              </span>
            </span>
            <span v-else>
              Environment changed
              <span v-if="detail.package">({{ detail.package }})</span>
            </span>
          </li>
        </ul>
      </div>

      <!-- Folded summary -->
      <div v-if="folded" class="cell-folded-summary" @click="folded = false">
        <span class="folded-source">{{ cell.source.split('\n')[0] || '(empty)' }}</span>
        <span v-if="cell.source.split('\n').length > 1" class="folded-lines">
          {{ cell.source.split('\n').length }} lines
        </span>
      </div>

      <!-- Rendered preview for markdown cells. Click anywhere to edit. -->
      <div
        v-if="!folded && cell.language === 'markdown' && isMarkdownPreviewing"
        class="markdown-preview"
        :class="{ empty: !renderedMarkdownSource }"
        title="Click to edit"
        @click="enterMarkdownEdit"
        v-html="renderedMarkdownSource || '<p class=\'placeholder\'>(empty markdown cell)</p>'"
      ></div>

      <div
        v-show="!folded && !(cell.language === 'markdown' && isMarkdownPreviewing)"
        ref="editorEl"
        class="editor-container"
        @focusout="
          cell.language === 'markdown' ? exitMarkdownEditOnBlur() : flushCellSource(cell.id)
        "
      />

      <!-- Console output (stdout/stderr) -->
      <div v-if="!folded && cellConsoleText(cell)" class="cell-console">
        <pre>{{
          outputExpanded || !isLongText(cellConsoleText(cell))
            ? cellConsoleText(cell)
            : compactText(cellConsoleText(cell))
        }}</pre>
        <button
          v-if="isLongText(cellConsoleText(cell))"
          class="output-toggle"
          @click="outputExpanded = !outputExpanded"
        >
          {{
            outputExpanded ? 'Show less' : `Show more (${overflowSummary(cellConsoleText(cell))})`
          }}
        </button>
      </div>

      <!-- Output -->
      <div v-if="!folded && cell.output" class="cell-output">
        <div v-if="cell.output.error" class="output-error">
          <div v-if="remoteExecutionIssueSummary" class="remote-error-summary">
            <span class="remote-error-label">Remote</span>
            <span>{{ remoteExecutionIssueSummary }}</span>
            <span v-if="lastRemoteWorkerLabel" class="remote-error-pill">
              {{ lastRemoteWorkerLabel }}
            </span>
            <span v-if="lastRemoteTransportLabel" class="remote-error-pill">
              {{ lastRemoteTransportLabel }}
            </span>
            <span v-if="lastRemoteBuildId" class="remote-error-pill">
              {{ lastRemoteBuildId }}
            </span>
            <span v-if="lastRemoteBuildState" class="remote-error-pill">
              {{ lastRemoteBuildState }}
            </span>
            <span v-if="lastRemoteErrorCode" class="remote-error-pill">
              {{ lastRemoteErrorCode }}
            </span>
            <span v-if="effectiveWorkerHealthLabel" class="remote-error-pill">
              {{ effectiveWorkerHealthLabel }}
            </span>
          </div>
          <pre class="output-error-detail">{{ cell.output.error }}</pre>
          <!--
            Install button is gated on language. ``uv add`` only works
            for Python packages; firing it for an R suggestion would
            install the wrong thing (or a Python package that happens
            to share the name) into the wrong env. R cells still see
            the structured hint as a read-only label so the user
            knows the missing package; the install action lands when
            the R env-job path ships.
          -->
          <div
            v-if="cell.suggestInstall && cell.suggestInstallLanguage === 'python'"
            class="suggest-install"
          >
            <span
              >Missing package <code>{{ cell.suggestInstall }}</code></span
            >
            <button
              class="btn-install"
              :disabled="installInProgress || environmentMutationActive"
              @click="installSuggestedPackage"
            >
              {{
                installInProgress
                  ? `Installing ${installTargetPackage}`
                  : `Install ${cell.suggestInstall}`
              }}
            </button>
          </div>
          <div
            v-else-if="cell.suggestInstall && cell.suggestInstallLanguage === 'r'"
            class="suggest-install suggest-install-readonly"
          >
            <span
              >Missing R package <code>{{ cell.suggestInstall }}</code> — run
              <code>install.packages("{{ cell.suggestInstall }}")</code> in an R cell to
              install.</span
            >
          </div>
        </div>
        <template v-else>
          <div
            v-for="(output, index) in visibleOutputs"
            :key="outputKey(output, index)"
            class="output-block"
          >
            <div v-if="output.rows?.length" class="output-table-wrap">
              <table class="output-table">
                <thead>
                  <tr>
                    <th v-for="col in output.columns" :key="col">{{ col }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, rowIndex) in output.rows?.slice(0, 50)" :key="rowIndex">
                    <td v-for="col in output.columns" :key="col">{{ row[col] }}</td>
                  </tr>
                </tbody>
              </table>
              <div v-if="(output.rowCount ?? 0) > 50" class="row-count">
                showing 50 of {{ output.rowCount?.toLocaleString() }} rows
              </div>
            </div>
            <div
              v-else-if="output.contentType === 'image/png' && output.inlineDataUrl"
              class="output-image"
            >
              <img
                :src="output.inlineDataUrl"
                alt="Cell output"
                :width="output.width || undefined"
                :height="output.height || undefined"
              />
            </div>
            <div
              v-else-if="output.contentType === 'text/markdown' && output.markdownText"
              class="output-markdown"
              :class="{ collapsed: !outputExpanded && isLongText(output.markdownText) }"
              v-html="renderedMarkdownOutput(output)"
            ></div>
            <button
              v-if="
                output.contentType === 'text/markdown' &&
                output.markdownText &&
                isLongText(output.markdownText)
              "
              class="output-toggle"
              @click="outputExpanded = !outputExpanded"
            >
              {{ outputExpanded ? 'Show less' : 'Show more' }}
            </button>
            <div v-else-if="output.scalar !== undefined" class="output-scalar">
              <pre
                v-if="isJsonLike(formatScalar(output.scalar))"
                v-html="
                  highlightJson(
                    outputExpanded || !isLongText(formatScalar(output.scalar))
                      ? formatScalar(output.scalar)
                      : compactText(formatScalar(output.scalar)),
                  )
                "
              ></pre>
              <pre v-else>{{
                outputExpanded || !isLongText(formatScalar(output.scalar))
                  ? formatScalar(output.scalar)
                  : compactText(formatScalar(output.scalar))
              }}</pre>
              <button
                v-if="isLongText(formatScalar(output.scalar))"
                class="output-toggle"
                @click="outputExpanded = !outputExpanded"
              >
                {{
                  outputExpanded
                    ? 'Show less'
                    : `Show more (${overflowSummary(formatScalar(output.scalar))})`
                }}
              </button>
            </div>
          </div>
        </template>
      </div>

      <div v-if="installCompleted && installTargetPackage" class="install-complete-hint">
        Installed <code>{{ installTargetPackage }}</code
        >. Re-run the cell to continue.
      </div>

      <!-- Inspect REPL panel -->
      <InspectPanel v-if="isInspecting" :cell-id="cell.id" />
    </div>
  </div>
</template>

<style scoped>
.cell {
  display: flex;
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  margin-bottom: 8px;
  background: var(--bg-elevated);
  transition: border-color 0.2s;
}
.cell:hover {
  border-color: var(--accent-lavender);
}
.cell.status-running {
  border-color: var(--accent-primary);
}
.cell.status-ready {
  border-color: var(--accent-success);
}
.cell.status-stale {
  border-color: var(--accent-warning);
}
.cell.status-error {
  border-color: var(--accent-danger);
}

.cell-gutter {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 4px;
  gap: 8px;
  min-width: 36px;
  border-right: 1px solid var(--border-subtle);
}
.status-dot {
  font-size: 14px;
  color: var(--text-muted);
}
.status-ready .status-dot {
  color: var(--accent-success);
}
.status-running .status-dot {
  color: var(--accent-primary);
  animation: pulse 1s infinite;
}
.status-stale .status-dot {
  color: var(--accent-warning);
}
.status-error .status-dot {
  color: var(--accent-danger);
}

@keyframes pulse {
  50% {
    opacity: 0.4;
  }
}

.cell-actions {
  display: flex;
  flex-direction: column;
  gap: 2px;
  opacity: 0;
  transition: opacity 0.15s;
}
.cell:hover .cell-actions {
  opacity: 1;
}
.cell-actions button {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 14px;
  padding: 2px 4px;
  border-radius: 3px;
}
.cell-actions button:hover {
  background: var(--bg-input);
  color: var(--text-primary);
}
.cell-actions button.active {
  color: var(--accent-primary);
  background: var(--tint-primary);
}

.cell-folded-summary {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  cursor: pointer;
  color: var(--text-muted);
  font-size: 12px;
  font-family: monospace;
  background: var(--bg-base);
  border-radius: 4px;
}

.cell-folded-summary:hover {
  background: var(--bg-elevated);
  color: var(--text-secondary);
}

.folded-source {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  min-width: 0;
}

.folded-lines {
  flex-shrink: 0;
  font-size: 10px;
  color: var(--border-strong);
}

.cell-prompt {
  border-color: var(--tint-primary-strong);
}

.cell-prompt .cell-lang {
  background: var(--tint-primary-strong);
  color: var(--accent-primary);
}

.cell-markdown {
  border-color: var(--border-subtle);
}

.cell-markdown .cell-lang {
  background: var(--bg-input);
  color: var(--text-secondary);
}

.markdown-preview {
  padding: 12px 16px;
  cursor: text;
  color: var(--text-primary);
  border-radius: 4px;
  min-height: 32px;
}

.markdown-preview:hover {
  background: var(--bg-elevated);
}

.markdown-preview.empty {
  color: var(--text-muted);
  font-style: italic;
}

.markdown-preview .placeholder {
  margin: 0;
}

.markdown-preview :deep(h1),
.markdown-preview :deep(h2),
.markdown-preview :deep(h3),
.markdown-preview :deep(h4),
.markdown-preview :deep(h5),
.markdown-preview :deep(h6) {
  margin: 0.6em 0 0.3em;
  font-weight: 600;
}

.markdown-preview :deep(h1) {
  font-size: 1.6em;
}
.markdown-preview :deep(h2) {
  font-size: 1.35em;
}
.markdown-preview :deep(h3) {
  font-size: 1.15em;
}
.markdown-preview :deep(p),
.markdown-preview :deep(ul),
.markdown-preview :deep(ol),
.markdown-preview :deep(blockquote),
.markdown-preview :deep(pre),
.markdown-preview :deep(table) {
  margin: 0.4em 0;
}

/* The global ``* { padding: 0 }`` reset wipes the browser default
 * ``padding-inline-start: 40px`` on lists, which collapses every nesting
 * level to the same column. Restore an explicit indent on each ``ul``/
 * ``ol`` so each level visually steps in. ``list-style-position: outside``
 * is the default but we set it explicitly because we're re-establishing
 * the layout from scratch.
 */
.markdown-preview :deep(ul),
.markdown-preview :deep(ol) {
  padding-inline-start: 1.6em;
  list-style-position: outside;
}

.markdown-preview :deep(li) {
  margin: 0.15em 0;
}

.markdown-preview :deep(code) {
  background: var(--bg-input);
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 0.9em;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

.markdown-preview :deep(pre) {
  background: var(--bg-input);
  padding: 8px 12px;
  border-radius: 4px;
  overflow-x: auto;
}

.markdown-preview :deep(pre code) {
  background: none;
  padding: 0;
}

.markdown-preview :deep(blockquote) {
  border-left: 3px solid var(--border-strong);
  padding-left: 12px;
  color: var(--text-secondary);
}

.markdown-preview :deep(table) {
  border-collapse: collapse;
}

.markdown-preview :deep(th),
.markdown-preview :deep(td) {
  border: 1px solid var(--border-subtle);
  padding: 4px 8px;
}

.markdown-preview :deep(a) {
  color: var(--accent-primary);
  text-decoration: underline;
}

.shadow-badge {
  font-size: 10px;
  padding: 1px 6px;
  background: var(--tint-warning);
  color: var(--accent-warning);
  border-radius: 4px;
  cursor: help;
}

.cell-body {
  flex: 1;
  min-width: 0;
}

.variant-tabs {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 12px 0;
  border-bottom: 1px solid var(--border-subtle);
}
.variant-tab {
  padding: 3px 10px;
  font-size: 11px;
  font-family: var(--font-mono, monospace);
  border: 1px solid var(--border-subtle);
  border-bottom: none;
  border-radius: 4px 4px 0 0;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
}
.variant-tab:hover {
  color: var(--text-default);
  background: var(--background-hover, rgba(127, 127, 127, 0.08));
}
.variant-tab.active {
  background: var(--background-elevated, var(--background-default));
  color: var(--text-default);
  border-color: var(--accent-primary, var(--border-strong));
  font-weight: 600;
}
.variant-tab-add {
  padding: 3px 8px;
  font-size: 13px;
  line-height: 1;
  font-weight: 600;
  color: var(--text-muted);
}
.variant-tab-add:hover {
  color: var(--accent-primary, var(--text-default));
}
.variant-group-label {
  margin-left: auto;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
}

.cell-meta {
  display: flex;
  flex-direction: column;
  padding: 4px 12px;
  font-size: 11px;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border-subtle);
  min-width: 0;
  gap: 2px;
}
.cell-meta-row {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}
.cell-meta-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
}
.cell-meta-runtime {
  flex-wrap: wrap;
  gap: 8px;
}
.cell-meta-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
  flex-shrink: 0;
}
.cell-lang {
  text-transform: uppercase;
  font-weight: 600;
}
.name-badge {
  background: var(--tint-primary);
  color: var(--accent-primary);
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 600;
}
.cell-vars code {
  color: var(--accent-primary);
}
.cache-badge {
  background: var(--tint-success);
  color: var(--accent-success);
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 600;
}
.loop-progress-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--tint-primary);
  color: var(--accent-primary);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
}
.loop-progress-badge.running {
  background: var(--tint-warning);
  color: var(--accent-warning);
  border: 1px solid var(--tint-warning);
}
.loop-progress-badge.done {
  background: var(--tint-success);
  color: var(--accent-success);
}
.loop-spinner {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  border: 1.5px solid currentColor;
  border-right-color: transparent;
  animation: loop-spin 0.9s linear infinite;
}
@keyframes loop-spin {
  to {
    transform: rotate(360deg);
  }
}

/* Plain Python cells: badge shown while the cell is running. Reuses
   the loop-progress-badge visual treatment so loop and non-loop
   running states look the same. The spinner animation is shared with
   loop-spin via the running-spinner rule below. */
.running-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--tint-warning);
  color: var(--accent-warning);
  border: 1px solid var(--tint-warning);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
}
.running-spinner {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  border: 1.5px solid currentColor;
  border-right-color: transparent;
  animation: loop-spin 0.9s linear infinite;
}

/* Cancel button replaces the run button while a cell is executing.
   Uses the warning color palette to match the running badge. */
.cell-actions .cancel-btn {
  color: var(--accent-warning);
}
.cell-actions .cancel-btn:hover {
  color: var(--accent-warning);
  background: var(--tint-warning);
}

.mount-badge {
  background: var(--tint-teal);
  color: var(--accent-teal);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
}
.worker-badge {
  background: var(--tint-mauve);
  color: var(--accent-mauve);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
}
.dispatch-badge {
  background: var(--tint-warning);
  color: var(--accent-warning);
  border: 1px solid var(--tint-warning);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
  animation: dispatch-pulse 1.2s ease-in-out infinite;
}
@keyframes dispatch-pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.55;
  }
}
.worker-transport-badge {
  background: var(--tint-info);
  color: var(--accent-info);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
}
.worker-health-badge {
  background: var(--tint-primary);
  color: var(--accent-primary);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
}
.worker-health-badge.warning {
  background: var(--tint-warning);
  color: var(--accent-warning);
}
.timeout-badge {
  background: var(--tint-warning);
  color: var(--accent-warning);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
}
.env-badge {
  background: var(--tint-peach);
  color: var(--accent-peach);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
}
.exec-method-badge {
  background: var(--tint-primary);
  color: var(--accent-primary);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
}
.leaf-badge {
  background: var(--tint-primary-strong);
  color: var(--accent-primary);
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 600;
}
.duration {
  white-space: nowrap;
}
.remote-execution-summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border-subtle);
  background: var(--bg-base);
}
.remote-execution-summary.tone-info {
  background: var(--bg-base);
}
.remote-execution-summary.tone-success {
  background: var(--tint-success);
}
.remote-execution-summary.tone-warning {
  background: var(--tint-warning);
}
.remote-execution-summary.tone-error {
  background: var(--tint-danger);
}
.remote-execution-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-secondary);
}
.remote-execution-detail {
  font-size: 12px;
  color: var(--text-primary);
}
.remote-execution-pill {
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--bg-input);
  color: var(--accent-primary);
  font-size: 11px;
  font-weight: 600;
}
.remote-execution-summary.tone-success .remote-execution-pill {
  color: var(--accent-success);
}
.remote-execution-summary.tone-warning .remote-execution-pill {
  color: var(--accent-warning);
}
.remote-execution-summary.tone-error .remote-execution-pill {
  color: var(--accent-danger);
}
.annotation-badge {
  background: var(--tint-warning);
  color: var(--accent-warning);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
}
.annotation-summary-chip {
  background: var(--tint-peach);
  color: var(--accent-peach);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
}
.annotation-diagnostic-badge {
  background: var(--tint-danger);
  color: var(--accent-danger);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
  cursor: help;
}

/* Module-cell marker — shown when the cell's source is pure enough to
 * be shared as a synthetic module across downstream cells. */
.module-cell-badge {
  background: var(--tint-primary);
  color: var(--accent-primary);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
  cursor: help;
  letter-spacing: 0.02em;
}

/* v1.1: Causality inspector */
.causality-btn {
  background: var(--tint-warning);
  color: var(--accent-warning);
  border: 1px solid var(--tint-warning);
  padding: 1px 8px;
  border-radius: 3px;
  font-size: 10px;
  cursor: pointer;
  font-weight: 600;
}
.causality-btn:hover {
  background: var(--tint-warning);
  border-color: var(--tint-warning);
}

@media (max-width: 900px) {
  .cell-meta-row {
    flex-direction: column;
    align-items: stretch;
  }

  .cell-meta-actions {
    margin-left: 0;
    justify-content: flex-end;
    flex-wrap: wrap;
  }
}

.causality-panel {
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-subtle);
  padding: 8px 12px;
  font-size: 12px;
}
.causality-header {
  color: var(--accent-warning);
  font-weight: 600;
  margin-bottom: 4px;
}
.causality-reason {
  background: var(--tint-warning);
  padding: 1px 6px;
  border-radius: 3px;
}
.causality-details {
  list-style: none;
  padding: 0;
  margin: 0;
}
.causality-detail {
  color: var(--text-secondary);
  padding: 2px 0;
  display: flex;
  align-items: center;
  gap: 6px;
}
.causality-detail code {
  color: var(--accent-primary);
}
.detail-icon {
  color: var(--accent-warning);
  font-size: 11px;
  width: 16px;
  text-align: center;
}
.version-change {
  color: var(--text-muted);
  font-size: 10px;
}

/* Console output */
.cell-console {
  border-top: 1px solid var(--border-subtle);
  padding: 6px 12px;
  font-size: 12px;
  background: var(--bg-base);
}
.cell-console pre {
  margin: 0;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  color: var(--text-secondary);
  white-space: pre-wrap;
}

.editor-container {
  /* CodeMirror auto-sizes to content; a forced min-height left ~18px of
     dead space below 1-line cells. The cell-gutter padding (8px top +
     bottom) already guarantees a visible click target on empty cells. */
}

.cell-output {
  border-top: 1px solid var(--border-subtle);
  padding: 8px 12px;
  font-size: 13px;
}
.output-block + .output-block {
  margin-top: 12px;
}
.output-error {
  color: var(--accent-danger);
}
.output-error-detail {
  margin: 0;
  font-family: monospace;
  white-space: pre-wrap;
}
.remote-error-summary {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
  color: var(--accent-warning);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px;
}
.remote-error-label,
.remote-error-pill {
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--bg-input);
  color: var(--accent-warning);
  font-size: 11px;
  font-weight: 700;
}
.suggest-install {
  margin-top: 8px;
  padding: 8px 10px;
  background: var(--tint-primary);
  border: 1px solid var(--tint-primary-strong);
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 12px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px;
  color: var(--text-primary);
}
.suggest-install code {
  color: var(--accent-primary);
  font-weight: 600;
}

/* R-only variant — no install button (action endpoint not yet wired).
   Slightly muted tint so it reads as advisory rather than actionable. */
.suggest-install-readonly {
  background: var(--bg-input);
  border-color: var(--border-subtle);
}
.btn-install {
  background: var(--accent-primary);
  color: var(--bg-elevated);
  border: none;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
  flex-shrink: 0;
}
.btn-install:hover {
  background: var(--accent-primary-hover);
}
.btn-install:disabled {
  background: var(--text-muted);
  color: var(--text-primary);
  cursor: wait;
}
.install-complete-hint {
  margin-top: 8px;
  padding: 8px 10px;
  background: var(--tint-success);
  border: 1px solid var(--tint-success);
  border-radius: 6px;
  color: var(--accent-success);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px;
}
.install-complete-hint code {
  color: var(--text-primary);
}
.output-table-wrap {
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
}
.output-image {
  overflow-x: auto;
}
.output-image img {
  display: block;
  max-width: 100%;
  height: auto;
  border: 1px solid var(--bg-input);
  border-radius: 6px;
  background: var(--bg-base);
}
.output-markdown {
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  line-height: 1.6;
}
.output-markdown :deep(h1),
.output-markdown :deep(h2),
.output-markdown :deep(h3),
.output-markdown :deep(h4),
.output-markdown :deep(h5),
.output-markdown :deep(h6) {
  margin: 0 0 8px;
  color: var(--cat-rosewater);
}
.output-markdown :deep(p),
.output-markdown :deep(blockquote),
.output-markdown :deep(pre),
.output-markdown :deep(table),
.output-markdown :deep(ul),
.output-markdown :deep(ol) {
  margin: 0 0 10px;
}

/* Mirror the .markdown-preview indent fix: the global ``* { padding: 0 }``
 * reset removes the browser default list indent, so without restoring
 * ``padding-inline-start`` every nesting level lands at column zero. */
.output-markdown :deep(ul),
.output-markdown :deep(ol) {
  padding-inline-start: 1.6em;
  list-style-position: outside;
}

.output-markdown :deep(li) {
  margin: 0.15em 0;
}
.output-markdown :deep(code) {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  background: var(--bg-base);
  color: var(--accent-warning);
  padding: 1px 4px;
  border-radius: 4px;
}
.output-markdown :deep(pre) {
  overflow-x: auto;
  padding: 10px 12px;
  border: 1px solid var(--bg-input);
  border-radius: 6px;
  background: var(--bg-base);
}
.output-markdown :deep(pre code) {
  padding: 0;
  background: transparent;
}
.output-markdown :deep(blockquote) {
  padding-left: 12px;
  border-left: 3px solid var(--accent-primary);
  color: var(--cat-subtext1);
}
.output-markdown :deep(a) {
  color: var(--accent-primary);
}
.output-markdown :deep(table) {
  width: 100%;
  border-collapse: collapse;
}
.output-markdown :deep(th),
.output-markdown :deep(td) {
  padding: 6px 10px;
  border: 1px solid var(--bg-input);
  text-align: left;
}
.output-markdown :deep(th) {
  color: var(--accent-primary);
  background: var(--bg-surface);
}
.output-table {
  width: 100%;
  border-collapse: collapse;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
}
.output-table th {
  text-align: left;
  padding: 4px 12px;
  color: var(--accent-primary);
  border-bottom: 1px solid var(--bg-input);
  font-weight: 600;
  position: sticky;
  top: 0;
  background: var(--bg-elevated);
}
.output-table td {
  padding: 3px 12px;
  color: var(--text-primary);
  border-bottom: 1px solid var(--bg-elevated);
}
.output-table tr:hover td {
  background: var(--bg-input);
}
.row-count {
  color: var(--text-muted);
  font-size: 11px;
  margin-top: 4px;
}

.output-scalar pre {
  margin: 0;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-all;
}
.output-scalar :deep(.json-key) {
  color: var(--accent-primary);
}
.output-scalar :deep(.json-string) {
  color: var(--accent-success);
}
.output-scalar :deep(.json-number) {
  color: var(--accent-peach);
}
.output-scalar :deep(.json-bool) {
  color: var(--accent-mauve);
}
.output-markdown.collapsed {
  max-height: 200px;
  overflow: hidden;
  mask-image: linear-gradient(to bottom, black 70%, transparent 100%);
  -webkit-mask-image: linear-gradient(to bottom, black 70%, transparent 100%);
}
.output-toggle {
  display: block;
  margin-top: 4px;
  padding: 2px 8px;
  background: none;
  border: 1px solid var(--bg-input);
  border-radius: 4px;
  color: var(--text-muted);
  font-size: 11px;
  cursor: pointer;
}
.output-toggle:hover {
  border-color: var(--accent-primary);
  color: var(--accent-primary);
}
</style>
