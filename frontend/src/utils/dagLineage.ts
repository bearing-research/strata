// What a selected DAG node connects to: everything upstream and downstream of it.

interface Edge {
  from: string
  to: string
}

export interface Lineage {
  ancestors: Set<string>
  descendants: Set<string>
}

function reach(start: string, next: Map<string, string[]>): Set<string> {
  const seen = new Set<string>()
  const stack = [start]
  while (stack.length) {
    for (const id of next.get(stack.pop() as string) ?? []) {
      if (!seen.has(id) && id !== start) {
        seen.add(id)
        stack.push(id)
      }
    }
  }
  return seen
}

export function dagLineage(selected: string, edges: Edge[]): Lineage {
  const upstream = new Map<string, string[]>()
  const downstream = new Map<string, string[]>()
  for (const { from, to } of edges) {
    upstream.set(to, [...(upstream.get(to) ?? []), from])
    downstream.set(from, [...(downstream.get(from) ?? []), to])
  }
  return { ancestors: reach(selected, upstream), descendants: reach(selected, downstream) }
}

/** False for an edge that skips the selection, such as an ancestor straight to a descendant. */
export function edgeOnLineage(edge: Edge, selected: string, lineage: Lineage): boolean {
  const { ancestors, descendants } = lineage
  const upstreamSide = ancestors.has(edge.from) && (edge.to === selected || ancestors.has(edge.to))
  const downstreamSide =
    descendants.has(edge.to) && (edge.from === selected || descendants.has(edge.from))
  return upstreamSide || downstreamSide
}
