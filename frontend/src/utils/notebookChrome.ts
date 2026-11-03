// What the notebook page's top bar and sidebar show, as plain decisions.
import type { EditorChrome } from './framed.ts'

/** Below this width the top bar moves its secondary actions into the overflow menu. */
export const NARROW_HEADER_QUERY = '(max-width: 1279px)'

export type OverflowAction = 'app' | 'rerunAll' | 'shortcuts' | 'artifacts' | 'logs' | 'delete'

/** The top bar's overflow menu, in order; a framed editor has no links or delete. */
export function overflowActions(chrome: EditorChrome, narrow: boolean): OverflowAction[] {
  const actions: OverflowAction[] = []
  if (narrow) {
    actions.push('rerunAll')
    if (chrome.pageLinks) actions.push('app')
    actions.push('shortcuts')
  }
  if (chrome.pageLinks) actions.push('artifacts', 'logs')
  if (chrome.deleteNotebook) actions.push('delete')
  return actions
}

export function connectionStatus(
  loading: boolean,
  connected: boolean,
): { state: 'connecting' | 'connected' | 'disconnected'; label: string } {
  if (loading) return { state: 'connecting', label: 'Connecting' }
  return connected
    ? { state: 'connected', label: 'Connected' }
    : { state: 'disconnected', label: 'Not connected' }
}

/** The drawer starts collapsed unless the user left it open. */
export function dagDrawerStartsCollapsed(saved: string | null): boolean {
  return saved !== '0'
}

/** Which settings panels leave the collapsed group, because they need the user's eye. */
export function settingsAttention(state: {
  mounts: number
  worker: string | null | undefined
  environmentBusy: boolean
  environmentFailed: boolean
}): { mounts: boolean; worker: boolean; environment: boolean } {
  return {
    mounts: state.mounts > 0,
    worker: Boolean(state.worker) && state.worker !== 'local',
    environment: state.environmentBusy || state.environmentFailed,
  }
}
