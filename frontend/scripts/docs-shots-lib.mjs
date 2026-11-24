/**
 * Browser helpers the docs capture scripts share: open a notebook over REST,
 * and load its page with the theme pinned, settled enough to photograph.
 */
import process from 'node:process'

export const VIEWPORT = { width: 1440, height: 900 }
export const SCALE = 2
export const THEMES = ['light', 'dark']
export const TIMEOUT_MS = 30000

export function argFor(flag, fallback) {
  const i = process.argv.indexOf(flag)
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback
}

export async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { 'content-type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${method} ${url} -> ${res.status}: ${await res.text()}`)
  return res.status === 204 ? null : res.json()
}

export async function openNotebook(baseUrl, path) {
  const data = await api('POST', `${baseUrl}/v1/notebooks/open`, { path })
  return { sessionId: data.session_id, cells: data.cells }
}

/** Load a notebook page with the theme pinned, settled enough to photograph. */
export async function loadNotebook(
  browser,
  baseUrl,
  sessionId,
  theme,
  viewport = VIEWPORT,
  { drawer = true, scale = SCALE } = {},
) {
  const context = await browser.newContext({ viewport, deviceScaleFactor: scale })
  await context.addInitScript(
    ([value, collapsed]) => {
      try {
        localStorage.setItem('strata.theme', value)
        // Most shots show the drawer, which a fresh profile opens collapsed.
        localStorage.setItem('strata:dagDrawerCollapsed', collapsed)
      } catch {
        // Storage disabled: the page falls back to the system preference.
      }
    },
    [theme, drawer ? '0' : '1'],
  )
  const page = await context.newPage()

  // The header's deployment mode fails closed: if GET
  // /v1/notebooks/{id}/workers fails (a 429 from the client rate limiter is
  // the easy way to get one), the UI treats a personal server as service mode
  // and never re-syncs. Waiting for the right mode is both the settle signal
  // and the guard against photographing that wrong state; a reload is enough
  // to recover, since the next request is not rate limited.
  const personal = page.locator('.header[data-mode="personal"]')
  let lastError = null
  for (let attempt = 0; attempt < 3; attempt += 1) {
    // reload() on retries, not goto(): the app is hash-routed, so navigating to
    // the identical URL is a no-op and the retry would re-check the same page.
    if (attempt === 0) {
      await page.goto(`${baseUrl}/#/notebook/${sessionId}`, { waitUntil: 'networkidle' })
    } else {
      await page.waitForTimeout(2000)
      await page.reload({ waitUntil: 'networkidle' })
    }
    try {
      await personal.waitFor({ timeout: TIMEOUT_MS })
      lastError = null
      break
    } catch (err) {
      lastError = err
    }
  }
  if (lastError) {
    throw new Error(
      `notebook ${sessionId} never reported personal mode (a failed /workers ` +
        `request leaves the header in service mode): ${lastError.message}`,
    )
  }
  await page.locator('.cell').first().waitFor({ timeout: TIMEOUT_MS })
  // One animation frame past the last mutation: charts and the DAG settle here.
  await page.waitForTimeout(600)
  return { context, page }
}
