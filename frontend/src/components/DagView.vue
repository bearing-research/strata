<script setup lang="ts">
import * as dagre from '@dagrejs/dagre'
import { computed, reactive, ref } from 'vue'
import { useNotebook } from '../stores/notebook'
import type { Cell, CellId } from '../types/notebook'

const { notebook, orderedCells, dagEdges } = useNotebook()

interface NodeLayout {
  id: CellId
  x: number
  y: number
  width: number
  label: string
  fullLabel: string
  status: Cell['status']
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
}

// Per-node widths so labels like "Fetch latest prices" (19 chars) don't
// truncate inside a cramped 140px box, while single-word cells still
// render compact. JetBrains Mono at 11px renders ~6.6px/char; padding
// covers the rounded corners plus a little breathing room. Labels that
// would exceed maxNodeWidth still get ellipsized — unbounded growth
// makes dagre emit lopsided layouts.
const nodeHeight = 36
const charWidthPx = 6.6
const nodePaddingPx = 20
const minNodeWidth = 110
const maxNodeWidth = 260
const maxLabelLen = Math.floor((maxNodeWidth - nodePaddingPx) / charWidthPx)

function widthForLabel(label: string): number {
  const estimated = label.length * charWidthPx + nodePaddingPx
  return Math.max(minNodeWidth, Math.min(maxNodeWidth, estimated))
}

// Layout using dagre
const layout = computed(() => {
  // Inactive variants aren't part of the executable graph — they have
  // no edges and would render as orphan nodes. Hide them; the active
  // variant gets a stacked-card visual indicating siblings exist.
  const cells = orderedCells.value.filter((c) => c.variantActive !== false)
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

  // Compute per-node width up front so dagre routes edges around the
  // actual rendered box, not a uniform 140px placeholder.
  const nodeWidths = new Map<CellId, number>()
  for (const c of cells) {
    const name = c.annotations?.name
    const defines = c.defines.length ? c.defines.join(', ') : null
    const rawLabel = name || defines || c.id.slice(0, 8)
    nodeWidths.set(c.id, widthForLabel(rawLabel))
  }

  // Add nodes
  for (const c of cells) {
    g.setNode(c.id, { width: nodeWidths.get(c.id) ?? minNodeWidth, height: nodeHeight })
  }

  // Add edges (dedupe by from+to since multiple variables can connect same pair)
  const edgeSet = new Set<string>()
  for (const e of dagEdges.value) {
    const key = `${e.from_cell_id}->${e.to_cell_id}`
    if (!edgeSet.has(key)) {
      edgeSet.add(key)
      g.setEdge(e.from_cell_id, e.to_cell_id)
    }
  }

  dagre.layout(g)

  // Build node layouts
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
      status: c.status,
      variantSiblings: groupSize > 1 ? groupSize - 1 : 0,
      variantGroup: c.variantGroup ?? null,
      variantName: c.variantName ?? null,
    }
    nodes.push(node)
    nodeMap.set(c.id, node)
  }

  // Build edge layouts with dagre's routed points. Multiple variables
  // flowing between the same (from, to) pair share a single routed
  // path — rendering one arrow per variable just stacks identical
  // labels at the same midpoint, which looks like the text has been
  // jammed into one illegible blob. Group by pair and join the
  // variable names so each edge gets one clean label like "x, y, z".
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
    })
  }

  return { nodes, edges }
})

const nodes = computed(() => layout.value.nodes)
const edges = computed(() => layout.value.edges)

// SVG dimensions from dagre layout bounds
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

// Pan and zoom state
const pan = reactive({ x: 0, y: 0 })
const zoom = ref(1)
const isPanning = ref(false)
const panStart = reactive({ x: 0, y: 0, panX: 0, panY: 0 })

const viewBox = computed(() => {
  const w = svgWidth.value / zoom.value
  const h = svgHeight.value / zoom.value
  return `${-pan.x / zoom.value} ${-pan.y / zoom.value} ${w} ${h}`
})

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const factor = e.deltaY > 0 ? 0.9 : 1.1
  const newZoom = Math.max(0.3, Math.min(3, zoom.value * factor))
  zoom.value = newZoom
}

function onMouseDown(e: MouseEvent) {
  if (e.button !== 0) return
  isPanning.value = true
  panStart.x = e.clientX
  panStart.y = e.clientY
  panStart.panX = pan.x
  panStart.panY = pan.y
}

function onMouseMove(e: MouseEvent) {
  if (!isPanning.value) return
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

function scrollToCell(cellId: CellId) {
  const el = document.querySelector(`[data-testid="notebook-cell"][data-cell-id="${cellId}"]`)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    // Brief highlight flash
    el.classList.add('dag-jump-highlight')
    setTimeout(() => el.classList.remove('dag-jump-highlight'), 1500)
  }
}

function statusStroke(status: Cell['status']): string {
  switch (status) {
    case 'ready':
      return 'var(--accent-success)'
    case 'running':
      return 'var(--accent-primary)'
    case 'stale':
      return 'var(--accent-warning)'
    case 'error':
      return 'var(--accent-danger)'
    default:
      return 'var(--text-muted)'
  }
}

// Translucent fill companion to statusStroke. The SVG rect can't do
// string-concatenated alpha (e.g. "var(--accent-success)" + "22") because
// the result isn't valid CSS and browsers fall back to black — which is
// why these nodes rendered dark in both themes before.
function statusFill(status: Cell['status']): string {
  switch (status) {
    case 'ready':
      return 'var(--tint-success)'
    case 'running':
      return 'var(--tint-primary)'
    case 'stale':
      return 'var(--tint-warning)'
    case 'error':
      return 'var(--tint-danger)'
    default:
      return 'var(--tint-muted)'
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
    // Use cubic bezier through the control points
    for (let i = 0; i < rest.length - 1; i += 2) {
      const cp = rest[i]
      const end = rest[i + 1] ?? rest[i]
      d += ` Q ${cp.x} ${cp.y}, ${end.x} ${end.y}`
    }
    // If odd number of remaining points, line to the last
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
      <button class="dag-reset" title="Reset zoom" @click.stop="resetView">Reset</button>
    </div>
    <div
      class="dag-viewport"
      @wheel="onWheel"
      @mousedown="onMouseDown"
      @mousemove="onMouseMove"
      @mouseup="onMouseUp"
      @mouseleave="onMouseUp"
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
          :d="edgePath(e.points)"
          fill="none"
          stroke="var(--text-muted)"
          stroke-width="1.5"
          marker-end="url(#arrow)"
        />

        <!-- Edge variable labels (at midpoint of edge) -->
        <text
          v-for="(e, i) in edges"
          :key="'el' + i"
          :x="e.points[Math.floor(e.points.length / 2)]?.x + 6"
          :y="e.points[Math.floor(e.points.length / 2)]?.y + 3"
          fill="var(--cat-surface2)"
          font-size="9"
          font-family="JetBrains Mono, Fira Code, monospace"
        >
          {{ e.variable }}
        </text>

        <!-- Nodes -->
        <g v-for="n in nodes" :key="n.id" class="dag-node" @dblclick.stop="scrollToCell(n.id)">
          <!-- Stacked-card hint for variant groups: two offset shadow
               rects behind the active node convey "there are alternatives". -->
          <template v-if="n.variantSiblings > 0">
            <rect
              :x="n.x - n.width / 2 + 6"
              :y="n.y - nodeHeight / 2 + 6"
              :width="n.width"
              :height="nodeHeight"
              rx="6"
              :fill="statusFill(n.status)"
              :stroke="statusStroke(n.status)"
              stroke-width="1"
              opacity="0.35"
            />
            <rect
              :x="n.x - n.width / 2 + 3"
              :y="n.y - nodeHeight / 2 + 3"
              :width="n.width"
              :height="nodeHeight"
              rx="6"
              :fill="statusFill(n.status)"
              :stroke="statusStroke(n.status)"
              stroke-width="1"
              opacity="0.65"
            />
          </template>
          <rect
            :x="n.x - n.width / 2"
            :y="n.y - nodeHeight / 2"
            :width="n.width"
            :height="nodeHeight"
            rx="6"
            :fill="statusFill(n.status)"
            :stroke="statusStroke(n.status)"
            stroke-width="1.5"
          />
          <text
            :x="n.x"
            :y="n.y + 4"
            text-anchor="middle"
            fill="var(--text-primary)"
            font-size="11"
            font-family="JetBrains Mono, Fira Code, monospace"
          >
            <title>
              {{
                n.variantSiblings > 0
                  ? `${n.fullLabel}\nVariant: ${n.variantName} (${n.variantSiblings + 1} options in ${n.variantGroup})`
                  : n.fullLabel
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
.dag-panel {
  background: transparent;
  min-width: 180px;
  overflow: hidden;
  height: 100%;
  display: flex;
  flex-direction: column;
}
/* Matches ProfilingPanel's .profiling-header so the two panes in the
 * Execution drawer read as siblings, not strangers. */
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
  /* Panning would otherwise highlight the node / edge labels as a
   * text selection — ugly and distracting. The viewport never has
   * meaningful selectable text; double-click jumps to a cell via a
   * handler, not via native selection. */
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
</style>
