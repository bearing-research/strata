<script setup lang="ts">
import * as dagre from '@dagrejs/dagre'
import { PanelBottom, PanelRight } from '@lucide/vue'
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { useNotebook } from '../stores/notebook'
import { CELL_STATES, cellState, type CellState } from '../utils/cellState'
import { LABEL_HEIGHT, labelAnchor, placeEdgeLabels } from '../utils/dagLabels'
import { dagLineage, edgeOnLineage } from '../utils/dagLineage'
import { DOCK_MIN_FIT_SCALE, panToCenter, viewBox as dagViewBox } from '../utils/dagViewport'
import type { CellId } from '../types/notebook'

const props = defineProps<{ docked?: boolean }>()
defineEmits<{ toggleDock: [] }>()

// Held by the page, so moving the graph between the drawer and the dock keeps it.
const selected = defineModel<CellId | null>('selected', { default: null })

const { notebook, orderedCells, dagEdges } = useNotebook()

interface NodeLayout {
  id: CellId
  x: number
  y: number
  width: number
  label: string
  fullLabel: string
  state: CellState
  /** When this cell is the active variant of a group, the count of
   * sibling variants. Drives the "stacked card" visual + tooltip. */
  variantSiblings: number
  variantGroup: string | null
  variantName: string | null
}

interface EdgeLayout {
  from: NodeLayout
  to: NodeLayout
  variable: string
  points: { x: number; y: number }[]
  /** Top-left of the variable label, or null when no free spot was found. */
  label: { x: number; y: number } | null
}

// Node width follows the label (estimated at ~6.6px/char at 11px) but is
// capped and ellipsized: unbounded widths make dagre lay out lopsided.
const nodeHeight = 36
const charWidthPx = 6.6
const nodeIconPx = 12
const nodePaddingPx = 36
const minNodeWidth = 110
const maxNodeWidth = 260
const maxLabelLen = Math.floor((maxNodeWidth - nodePaddingPx) / charWidthPx)

function widthForLabel(label: string): number {
  const estimated = label.length * charWidthPx + nodePaddingPx
  return Math.max(minNodeWidth, Math.min(maxNodeWidth, estimated))
}

const layout = computed(() => {
  // Inactive variants show as the active node's stacked card; markdown cells
  // have no edges and render in the Notes panel instead.
  const cells = orderedCells.value.filter(
    (c) => c.variantActive !== false && c.language !== 'markdown',
  )
  if (cells.length === 0) return { nodes: [] as NodeLayout[], edges: [] as EdgeLayout[] }

  const variantGroupSizes = new Map<string, number>()
  for (const g of notebook.variantGroups) {
    variantGroupSizes.set(g.group, g.members.length)
  }

  const g = new dagre.graphlib.Graph()
  g.setGraph({
    rankdir: 'TB',
    nodesep: 24,
    ranksep: 48,
    marginx: 24,
    marginy: 24,
  })
  g.setDefaultEdgeLabel(() => ({}))

  // Up front, so dagre routes edges around the real box.
  const nodeWidths = new Map<CellId, number>()
  for (const c of cells) {
    const name = c.annotations?.name
    const defines = c.defines.length ? c.defines.join(', ') : null
    const rawLabel = name || defines || c.id.slice(0, 8)
    nodeWidths.set(c.id, widthForLabel(rawLabel))
  }

  for (const c of cells) {
    g.setNode(c.id, { width: nodeWidths.get(c.id) ?? minNodeWidth, height: nodeHeight })
  }

  // Dedupe by from+to: several variables can connect the same pair.
  const edgeSet = new Set<string>()
  for (const e of dagEdges.value) {
    const key = `${e.from_cell_id}->${e.to_cell_id}`
    if (!edgeSet.has(key)) {
      edgeSet.add(key)
      g.setEdge(e.from_cell_id, e.to_cell_id)
    }
  }

  dagre.layout(g)

  const nodeMap = new Map<CellId, NodeLayout>()
  const nodes: NodeLayout[] = []
  for (const c of cells) {
    const dagreNode = g.node(c.id)
    if (!dagreNode) continue

    const name = c.annotations?.name
    const defines = c.defines.length ? c.defines.join(', ') : null
    const rawLabel = name || defines || c.id.slice(0, 8)
    const label =
      rawLabel.length > maxLabelLen ? rawLabel.slice(0, maxLabelLen - 1) + '\u2026' : rawLabel

    const groupSize = c.variantGroup ? (variantGroupSizes.get(c.variantGroup) ?? 0) : 0
    const node: NodeLayout = {
      id: c.id,
      x: dagreNode.x,
      y: dagreNode.y,
      width: nodeWidths.get(c.id) ?? minNodeWidth,
      label,
      fullLabel: rawLabel,
      state: cellState(c),
      variantSiblings: groupSize > 1 ? groupSize - 1 : 0,
      variantGroup: c.variantGroup ?? null,
      variantName: c.variantName ?? null,
    }
    nodes.push(node)
    nodeMap.set(c.id, node)
  }

  // One edge per (from, to) pair labelled "x, y, z"; per-variable arrows
  // would stack their labels illegibly at the same midpoint.
  const edgeGroups = new Map<string, { from: NodeLayout; to: NodeLayout; vars: string[] }>()
  for (const e of dagEdges.value) {
    const from = nodeMap.get(e.from_cell_id)
    const to = nodeMap.get(e.to_cell_id)
    if (!from || !to) continue
    const key = `${e.from_cell_id}->${e.to_cell_id}`
    const existing = edgeGroups.get(key)
    if (existing) {
      if (!existing.vars.includes(e.variable)) existing.vars.push(e.variable)
    } else {
      edgeGroups.set(key, { from, to, vars: [e.variable] })
    }
  }

  const edges: EdgeLayout[] = []
  for (const [, group] of edgeGroups) {
    const dagreEdge = g.edge(group.from.id, group.to.id)
    const points = dagreEdge?.points ?? [
      { x: group.from.x, y: group.from.y + nodeHeight / 2 },
      { x: group.to.x, y: group.to.y - nodeHeight / 2 },
    ]
    edges.push({
      from: group.from,
      to: group.to,
      variable: group.vars.join(', '),
      points,
      label: null,
    })
  }

  const labelSpots = placeEdgeLabels(
    edges.map((e) => {
      const mid = e.points[Math.floor(e.points.length / 2)]
      return labelAnchor(mid, e.from.x, e.to.x, e.variable)
    }),
    nodes.map((n) => ({
      x: n.x - n.width / 2,
      y: n.y - nodeHeight / 2,
      width: n.width,
      height: nodeHeight,
    })),
  )
  edges.forEach((e, i) => {
    e.label = labelSpots[i]
  })

  return { nodes, edges }
})

const nodes = computed(() => layout.value.nodes)
const edges = computed(() => layout.value.edges)

// A selected cell that was deleted or left the graph selects nothing.
const activeSelection = computed(() =>
  nodes.value.some((n) => n.id === selected.value) ? selected.value : null,
)
const lineage = computed(() =>
  activeSelection.value
    ? dagLineage(
        activeSelection.value,
        edges.value.map((e) => ({ from: e.from.id, to: e.to.id })),
      )
    : null,
)

function nodeDimmed(id: CellId): boolean {
  if (!lineage.value) return false
  return (
    id !== activeSelection.value &&
    !lineage.value.ancestors.has(id) &&
    !lineage.value.descendants.has(id)
  )
}

function edgeDimmed(e: EdgeLayout): boolean {
  if (!lineage.value || !activeSelection.value) return false
  return !edgeOnLineage({ from: e.from.id, to: e.to.id }, activeSelection.value, lineage.value)
}

const svgWidth = computed(() => {
  if (nodes.value.length === 0) return 240
  let maxX = 0
  for (const n of nodes.value) maxX = Math.max(maxX, n.x + n.width / 2)
  return maxX + 24
})

const svgHeight = computed(() => {
  if (nodes.value.length === 0) return 100
  let maxY = 0
  for (const n of nodes.value) maxY = Math.max(maxY, n.y + nodeHeight / 2)
  return maxY + 24
})

const pan = reactive({ x: 0, y: 0 })
const zoom = ref(1)
const isPanning = ref(false)
const panStart = reactive({ x: 0, y: 0, panX: 0, panY: 0 })
// A click ends every pan, so one that moved doesn't select.
const panned = ref(false)
const PAN_SLOP_PX = 3

const viewportEl = ref<HTMLElement | null>(null)
const panel = reactive({ width: 0, height: 0 })

const minFitScale = computed(() => (props.docked ? DOCK_MIN_FIT_SCALE : undefined))

const viewBox = computed(() => {
  // Before the first measure, the whole graph.
  if (!panel.width || !panel.height) return `0 0 ${svgWidth.value} ${svgHeight.value}`
  const box = dagViewBox({
    graphWidth: svgWidth.value,
    graphHeight: svgHeight.value,
    panelWidth: panel.width,
    panelHeight: panel.height,
    zoom: zoom.value,
    panX: pan.x,
    panY: pan.y,
    minFitScale: minFitScale.value,
  })
  return `${box.x} ${box.y} ${box.width} ${box.height}`
})

// Moving to the dock remounts this, and its narrower panel can leave the selection off screen.
// The drawer shows the whole graph, so panning there would only push part of it out of view.
function centerSelection() {
  const node = nodes.value.find((n) => n.id === activeSelection.value)
  if (!node) return
  const { panX, panY } = panToCenter(
    {
      graphWidth: svgWidth.value,
      graphHeight: svgHeight.value,
      panelWidth: panel.width,
      panelHeight: panel.height,
      zoom: zoom.value,
      minFitScale: minFitScale.value,
    },
    node,
  )
  pan.x = panX
  pan.y = panY
}

let panelObserver: ResizeObserver | null = null
onMounted(() => {
  if (!viewportEl.value) return
  let first = true
  panelObserver = new ResizeObserver(([entry]) => {
    panel.width = entry.contentRect.width
    panel.height = entry.contentRect.height
    if (first && props.docked) centerSelection()
    first = false
  })
  panelObserver.observe(viewportEl.value)
})
onUnmounted(() => panelObserver?.disconnect())

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const factor = e.deltaY > 0 ? 0.9 : 1.1
  const newZoom = Math.max(0.3, Math.min(3, zoom.value * factor))
  zoom.value = newZoom
}

function onMouseDown(e: MouseEvent) {
  if (e.button !== 0) return
  isPanning.value = true
  panned.value = false
  panStart.x = e.clientX
  panStart.y = e.clientY
  panStart.panX = pan.x
  panStart.panY = pan.y
}

function onMouseMove(e: MouseEvent) {
  if (!isPanning.value) return
  if (
    Math.abs(e.clientX - panStart.x) > PAN_SLOP_PX ||
    Math.abs(e.clientY - panStart.y) > PAN_SLOP_PX
  ) {
    panned.value = true
  }
  pan.x = panStart.panX + (e.clientX - panStart.x)
  pan.y = panStart.panY + (e.clientY - panStart.y)
}

function onMouseUp() {
  isPanning.value = false
}

function resetView() {
  pan.x = 0
  pan.y = 0
  zoom.value = 1
}

function selectNode(cellId: CellId) {
  selected.value = cellId
  scrollToCell(cellId)
}

function onNodeClick(cellId: CellId) {
  if (!panned.value) selectNode(cellId)
}

function clearSelection() {
  if (!panned.value) selected.value = null
}

function scrollToCell(cellId: CellId) {
  const el = document.querySelector(`[data-testid="notebook-cell"][data-cell-id="${cellId}"]`)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    el.classList.add('dag-jump-highlight')
    setTimeout(() => el.classList.remove('dag-jump-highlight'), 1500)
  }
}

/** Build a smooth path through dagre's edge points */
function edgePath(points: { x: number; y: number }[]): string {
  if (points.length < 2) return ''
  const [start, ...rest] = points
  let d = `M ${start.x} ${start.y}`
  if (rest.length === 1) {
    d += ` L ${rest[0].x} ${rest[0].y}`
  } else {
    for (let i = 0; i < rest.length - 1; i += 2) {
      const cp = rest[i]
      const end = rest[i + 1] ?? rest[i]
      d += ` Q ${cp.x} ${cp.y}, ${end.x} ${end.y}`
    }
    if (rest.length % 2 === 1) {
      d += ` L ${rest[rest.length - 1].x} ${rest[rest.length - 1].y}`
    }
  }
  return d
}
</script>

<template>
  <div class="dag-panel">
    <div class="dag-header">
      <span>Cell DAG</span>
      <div class="dag-actions">
        <button class="dag-reset" title="Reset zoom" @click.stop="resetView">Reset</button>
        <button
          class="dag-reset dag-dock-toggle"
          data-testid="dag-dock-toggle"
          :aria-label="
            docked ? 'Move the graph to the bottom drawer' : 'Dock the graph on the right'
          "
          :title="docked ? 'Move the graph to the bottom drawer' : 'Dock the graph on the right'"
          @click.stop="$emit('toggleDock')"
        >
          <PanelBottom v-if="docked" /><PanelRight v-else />
        </button>
      </div>
    </div>
    <div
      ref="viewportEl"
      class="dag-viewport"
      @wheel="onWheel"
      @mousedown="onMouseDown"
      @mousemove="onMouseMove"
      @mouseup="onMouseUp"
      @mouseleave="onMouseUp"
      @click="clearSelection"
    >
      <svg
        width="100%"
        height="100%"
        :viewBox="viewBox"
        preserveAspectRatio="xMidYMid meet"
        class="dag-svg"
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-muted)" />
          </marker>
        </defs>

        <!-- Edges -->
        <path
          v-for="(e, i) in edges"
          :key="'e' + i"
          :class="{ 'dag-dimmed': edgeDimmed(e) }"
          :d="edgePath(e.points)"
          fill="none"
          stroke="var(--text-muted)"
          stroke-width="1.5"
          marker-end="url(#arrow)"
        >
          <title>{{ e.variable }}</title>
        </path>

        <!-- Edge variable labels (by the edge's midpoint, clear of nodes and each other) -->
        <template v-for="(e, i) in edges" :key="'el' + i">
          <text
            v-if="e.label"
            :class="{ 'dag-dimmed': edgeDimmed(e) }"
            :x="e.label.x"
            :y="e.label.y + LABEL_HEIGHT - 2"
            fill="var(--cat-surface2)"
            font-size="9"
            font-family="JetBrains Mono, Fira Code, monospace"
          >
            {{ e.variable }}
          </text>
        </template>

        <!-- Nodes -->
        <g
          v-for="n in nodes"
          :key="n.id"
          class="dag-node"
          :class="[
            `state-${n.state}`,
            { 'dag-selected': n.id === activeSelection, 'dag-dimmed': nodeDimmed(n.id) },
          ]"
          :data-testid="`dag-node-${n.id}`"
          role="button"
          tabindex="0"
          :aria-label="`Go to ${n.fullLabel}, ${CELL_STATES[n.state].label}, and show what it connects to`"
          :aria-pressed="n.id === activeSelection"
          @click.stop="onNodeClick(n.id)"
          @keydown.enter.prevent="selectNode(n.id)"
        >
          <!-- Stacked-card hint: the group has other variants -->
          <template v-if="n.variantSiblings > 0">
            <rect
              :x="n.x - n.width / 2 + 6"
              :y="n.y - nodeHeight / 2 + 6"
              :width="n.width"
              :height="nodeHeight"
              rx="6"
              fill="var(--state-tint)"
              stroke="var(--state)"
              stroke-width="1"
              opacity="0.35"
            />
            <rect
              :x="n.x - n.width / 2 + 3"
              :y="n.y - nodeHeight / 2 + 3"
              :width="n.width"
              :height="nodeHeight"
              rx="6"
              fill="var(--state-tint)"
              stroke="var(--state)"
              stroke-width="1"
              opacity="0.65"
            />
          </template>
          <rect
            class="dag-node-box"
            :x="n.x - n.width / 2"
            :y="n.y - nodeHeight / 2"
            :width="n.width"
            :height="nodeHeight"
            rx="6"
            fill="var(--state-tint)"
            stroke="var(--state)"
            stroke-width="1.5"
          />
          <component
            :is="CELL_STATES[n.state].icon"
            class="dag-node-icon"
            :x="n.x - n.width / 2 + 8"
            :y="n.y - nodeIconPx / 2"
            :width="nodeIconPx"
            :height="nodeIconPx"
            :fill="n.state === 'ready' ? 'currentColor' : 'none'"
          />
          <text
            :x="n.x + nodeIconPx / 2"
            :y="n.y + 4"
            text-anchor="middle"
            fill="var(--text-primary)"
            font-size="11"
          >
            <title>
              {{
                n.variantSiblings > 0
                  ? `${n.fullLabel}\n${CELL_STATES[n.state].label}\nVariant: ${n.variantName} (${n.variantSiblings + 1} options in ${n.variantGroup})`
                  : `${n.fullLabel}\n${CELL_STATES[n.state].label}`
              }}
            </title>
            {{ n.label }}
          </text>
          <!-- Variant badge: small chip in the top-right corner -->
          <g v-if="n.variantSiblings > 0">
            <rect
              :x="n.x + n.width / 2 - 36"
              :y="n.y - nodeHeight / 2 - 6"
              width="34"
              height="14"
              rx="3"
              fill="var(--accent-primary, #4c8bf5)"
              opacity="0.9"
            />
            <text
              :x="n.x + n.width / 2 - 19"
              :y="n.y - nodeHeight / 2 + 4"
              text-anchor="middle"
              fill="white"
              font-size="9"
              font-family="JetBrains Mono, Fira Code, monospace"
              font-weight="600"
            >
              {{ n.variantName }}
            </text>
          </g>
        </g>
      </svg>
    </div>
  </div>
</template>

<style scoped>
/* A nested svg takes its size from attributes in Chrome and from CSS elsewhere: .lucide is 1em. */
.dag-node-icon {
  font-size: 12px;
  color: var(--state);
}
.dag-panel {
  background: transparent;
  min-width: 180px;
  overflow: hidden;
  height: 100%;
  display: flex;
  flex-direction: column;
}
/* Matches ProfilingPanel's .profiling-header (sibling panes). */
.dag-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 1px solid var(--border-subtle);
}
.dag-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}
.dag-dock-toggle {
  display: inline-flex;
  align-items: center;
  font-size: 12px;
}
.dag-reset {
  font-size: 10px;
  padding: 2px 6px;
  background: none;
  border: 1px solid var(--bg-input);
  border-radius: 4px;
  color: var(--text-muted);
  cursor: pointer;
  text-transform: none;
  letter-spacing: 0;
}
.dag-reset:hover {
  border-color: var(--accent-primary);
  color: var(--accent-primary);
}
.dag-viewport {
  overflow: hidden;
  cursor: grab;
  flex: 1;
  min-height: 0;
  /* Panning would otherwise select label text. */
  user-select: none;
  -webkit-user-select: none;
}
.dag-viewport:active {
  cursor: grabbing;
}
.dag-svg {
  display: block;
  width: 100%;
  height: 100%;
}
.dag-node {
  cursor: pointer;
}
.dag-node:hover rect {
  filter: brightness(1.3);
}
.dag-node:focus-visible {
  outline: none;
}
.dag-node:focus-visible .dag-node-box,
.dag-selected .dag-node-box {
  stroke-width: 3;
}
.dag-dimmed {
  opacity: 0.25;
}
</style>
