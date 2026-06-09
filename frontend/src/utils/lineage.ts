// Adapt the backend's flat lineage graph (nodes + edges) into the recursive
// tree the UI renders: model ← features ← scan ← table @ snapshot. Edges run
// from_uri (input) → to_uri (consumer), so a node's children are the from_uris
// of edges pointing at it.

import type { LineageGraph } from '../composables/useStrata'

export interface LineageTreeNode {
  uri: string
  label: string
  type: string // 'artifact' | 'table'
  version: number | null
  children: LineageTreeNode[]
}

function tableLabel(uri: string): string {
  // file:///wh#nyc.trips@snapshot=123  →  nyc.trips @ snapshot 123
  const hashIdx = uri.indexOf('#')
  let rest = hashIdx >= 0 ? uri.slice(hashIdx + 1) : uri
  const snapIdx = rest.indexOf('@snapshot=')
  if (snapIdx >= 0) {
    const table = rest.slice(0, snapIdx)
    const snap = rest.slice(snapIdx + '@snapshot='.length)
    return `${table} @ snapshot ${snap}`
  }
  return rest
}

export function lineageToTree(graph: LineageGraph, rootUri: string): LineageTreeNode {
  const nodeByUri = new Map(graph.nodes.map((n) => [n.uri, n]))
  const inputsOf = new Map<string, string[]>()
  for (const e of graph.edges) {
    const arr = inputsOf.get(e.to_uri)
    if (arr) arr.push(e.from_uri)
    else inputsOf.set(e.to_uri, [e.from_uri])
  }

  const seen = new Set<string>()
  function build(uri: string): LineageTreeNode {
    seen.add(uri)
    const n = nodeByUri.get(uri)
    const type = n?.type ?? 'artifact'
    const label =
      type === 'table' ? tableLabel(uri) : (n?.transform_ref ?? 'artifact')
    const children = (inputsOf.get(uri) ?? [])
      .filter((u) => !seen.has(u))
      .map((u) => build(u))
    return { uri, label, type, version: n?.version ?? null, children }
  }
  return build(rootUri)
}

/** Flatten the tree to indented rows for simple list rendering. */
export function flattenLineage(root: LineageTreeNode): Array<LineageTreeNode & { depth: number }> {
  const rows: Array<LineageTreeNode & { depth: number }> = []
  function walk(node: LineageTreeNode, depth: number) {
    rows.push({ ...node, depth })
    for (const c of node.children) walk(c, depth + 1)
  }
  walk(root, 0)
  return rows
}
