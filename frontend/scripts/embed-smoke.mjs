/**
 * Browser E2E smoke for the embeddable app view.
 *
 * Drives the real feature end to end against a running Strata notebook
 * server: create a notebook with a widget cell over REST, then load a
 * same-origin host page that embeds the app view in an <iframe> with
 * `?embed=1` and assert the embed contract:
 *
 *   1. chromeless    — the standalone header (title + Edit link) is gone
 *   2. content lives — the widget control panel renders inside the frame
 *   3. auto-resize   — the embed posts `strata:embed:resize` to the parent,
 *                      which grows the iframe past its seed height
 *
 * The host page is served same-origin (via Playwright request routing), so
 * it works under the default `frame-ancestors 'self'` — no special server
 * config needed. Exits non-zero on any failed assertion.
 *
 *   node scripts/embed-smoke.mjs --base-url http://127.0.0.1:8770
 */
import process from 'node:process'
import { chromium } from 'playwright'

const DEFAULT_BASE_URL = 'http://127.0.0.1:8770'
const TIMEOUT_MS = 20000

function argFor(flag, fallback) {
  const i = process.argv.indexOf(flag)
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${url} -> ${res.status}: ${await res.text()}`)
  return res.json()
}

async function putJson(url, body) {
  const res = await fetch(url, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`PUT ${url} -> ${res.status}: ${await res.text()}`)
  return res.json()
}

async function main() {
  const baseUrl = (argFor('--base-url', DEFAULT_BASE_URL) || DEFAULT_BASE_URL).replace(/\/$/, '')
  const parentPath = argFor('--parent-path', '/tmp')
  const stamp = argFor('--stamp', String(Date.now()))

  // 1. Create a notebook with a widget cell — all over the real REST API.
  const created = await postJson(`${baseUrl}/v1/notebooks/create`, {
    parent_path: parentPath,
    name: `embed_smoke_${stamp}`,
  })
  const sessionId = created.session_id
  if (!sessionId) throw new Error(`create returned no session_id: ${JSON.stringify(created)}`)

  const cell = await postJson(`${baseUrl}/v1/notebooks/${sessionId}/cells`, { language: 'widget' })
  const cellId = cell.id
  if (!cellId) throw new Error(`add cell returned no id: ${JSON.stringify(cell)}`)
  await putJson(`${baseUrl}/v1/notebooks/${sessionId}/cells/${cellId}`, {
    source: 'alpha = slider(0, 1, default=0.5)\n',
  })

  // 2. A same-origin host page that iframes the embed and records resizes.
  const embedUrl = `${baseUrl}/#/app/${sessionId}?embed=1`
  const hostUrl = `${baseUrl}/__embed_smoke_host__`
  const hostHtml = `<!doctype html><meta charset="utf-8"><body>
    <iframe id="nb" src="${embedUrl}" style="width:100%;border:0;height:120px"></iframe>
    <script>
      window.__resizes = []
      addEventListener('message', (e) => {
        if (e.data && e.data.type === 'strata:embed:resize') {
          window.__resizes.push(e.data.height)
          document.getElementById('nb').style.height = e.data.height + 'px'
        }
      })
    </script></body>`

  const browser = await chromium.launch({ headless: true })
  const failures = []
  try {
    const page = await browser.newPage()
    await page.route(hostUrl, (route) =>
      route.fulfill({ contentType: 'text/html', body: hostHtml }),
    )
    await page.goto(hostUrl, { waitUntil: 'load', timeout: TIMEOUT_MS })

    const iframeEl = await page.waitForSelector('#nb', { timeout: TIMEOUT_MS })
    const frame = await iframeEl.contentFrame()
    if (!frame) throw new Error('iframe has no content frame')

    // 1. embed mode rendered chromeless
    await frame.waitForSelector('.app-view.embed', { timeout: TIMEOUT_MS })
    const headerCount = await frame.locator('.app-header').count()
    if (headerCount !== 0)
      failures.push(`expected no .app-header in embed mode, found ${headerCount}`)

    // 2. the widget control panel is live inside the frame
    const widgetCount = await frame.locator('.widget-cell').count()
    if (widgetCount < 1) failures.push('widget control panel did not render inside the iframe')

    // 3. the embed posted a resize and the host grew the iframe past its seed
    await page.waitForFunction(() => (window.__resizes || []).length > 0, { timeout: TIMEOUT_MS })
    const height = await page.$eval('#nb', (el) => el.offsetHeight)
    if (!(height > 120)) failures.push(`iframe did not grow past seed height (got ${height})`)
  } finally {
    await browser.close()
  }

  if (failures.length) {
    console.error('EMBED SMOKE: FAIL')
    for (const f of failures) console.error('  - ' + f)
    process.exit(1)
  }
  console.log('EMBED SMOKE: PASS (chromeless + live widget + auto-resize)')
}

main().catch((err) => {
  console.error('EMBED SMOKE: ERROR', err)
  process.exit(1)
})
