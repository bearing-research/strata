// The notebook editor framed by a platform that already has the chrome
// (`?framed=1`, e.g. `#/notebook/<id>?framed=1`): its brand and the notebook's
// name are the parent page's, links to Strata's other pages would navigate the
// frame away from the notebook, the deployment-mode badge names nothing the
// member chose, and deleting the notebook is the platform's to offer, so its
// records see it. Everything that works on the notebook stays.

export interface EditorChrome {
  brand: boolean
  pageLinks: boolean
  modeBadge: boolean
  deleteNotebook: boolean
}

const FULL: EditorChrome = { brand: true, pageLinks: true, modeBadge: true, deleteNotebook: true }
const FRAMED: EditorChrome = {
  brand: false,
  pageLinks: false,
  modeBadge: false,
  deleteNotebook: false,
}

function isFramed(query: Record<string, unknown>): boolean {
  const raw = query.framed
  const value = Array.isArray(raw) ? raw[0] : raw
  return value === '1' || value === 'true'
}

/** Which parts of the editor's header to show, from the route's query. */
export function editorChrome(query: Record<string, unknown>): EditorChrome {
  return isFramed(query) ? FRAMED : FULL
}

/** The query for the editor reopened at `path`; a framed editor stays framed. */
export function editorQuery(
  current: Record<string, unknown>,
  path: string,
): Record<string, string> {
  return isFramed(current) ? { path, framed: '1' } : { path }
}
