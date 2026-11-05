// What part of the DAG a panel shows. A graph that fits is fitted. The drawer is wide, so it always
// fits the whole graph; the dock is narrow, so a wider graph stays at a scale its labels can be read
// at, and is panned, since an unreadable overview navigates nothing.
export const DOCK_MIN_FIT_SCALE = 0.8

export interface DagView {
  graphWidth: number
  graphHeight: number
  panelWidth: number
  panelHeight: number
  zoom: number
  panX: number
  panY: number
  /** Omitted: fit the whole graph, however small that makes it. */
  minFitScale?: number
}

export function viewScale(v: DagView): number {
  const fit = Math.min(v.panelWidth / v.graphWidth, v.panelHeight / v.graphHeight)
  return Math.max(fit, v.minFitScale ?? 0) * v.zoom
}

/** Centered across, and from the top when the graph is taller than the panel. */
export function viewBox(v: DagView): { x: number; y: number; width: number; height: number } {
  const scale = viewScale(v)
  const width = v.panelWidth / scale
  const height = v.panelHeight / scale
  return {
    x: (v.graphWidth - width) / 2 - v.panX / scale,
    y: (height > v.graphHeight ? (v.graphHeight - height) / 2 : 0) - v.panY / scale,
    width,
    height,
  }
}

/** The pan that puts a graph point at the center of the panel. */
export function panToCenter(
  v: Omit<DagView, 'panX' | 'panY'>,
  point: { x: number; y: number },
): { panX: number; panY: number } {
  const home = viewBox({ ...v, panX: 0, panY: 0 })
  const scale = viewScale({ ...v, panX: 0, panY: 0 })
  return {
    panX: (home.x + home.width / 2 - point.x) * scale,
    panY: (home.y + home.height / 2 - point.y) * scale,
  }
}
