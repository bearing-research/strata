/**
 * Browser smoke for the Artifacts page, which shows the server's store and each notebook's. Moving
 * between their routes keeps the page mounted, so it has to start over each time:
 *
 *   1. from the server store, a notebook's store shows that notebook's name, rows and totals
 *   2. from one notebook's store, another's shows only its own
 *   3. a slow load for the store just left never lands over the one now shown
 *   4. while the next store loads, the last one's rows and totals are gone
 *
 * Exits non-zero on any failed assertion.
 *
 *   node scripts/store-page-smoke.mjs --base-url http://127.0.0.1:8770 --parent-path /tmp
 */
import process from 'node:process'
import { chromium } from 'playwright'

const DEFAULT_BASE_URL = 'http://127.0.0.1:8770'
const TIMEOUT_MS = 20000
// A new notebook's first environment sync.
const SYNC_TIMEOUT_MS = 180000
// Longer than the page takes to load the next store.
const SLOW_MS = 2000

function argFor(flag, fallback) {
  const i = process.argv.indexOf(flag)
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

async function request(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${method} ${url} -> ${res.status}: ${await res.text()}`)
  return res.json()
}

async function createNotebook(baseUrl, parentPath, name, sources) {
  const { session_id: sessionId } = await request('POST', `${baseUrl}/v1/notebooks/create`, {
    parent_path: parentPath,
    name,
  })
  const cellIds = []
  for (const source of sources) {
    const { id } = await request('POST', `${baseUrl}/v1/notebooks/${sessionId}/cells`, {
      language: 'python',
    })
    await request('PUT', `${baseUrl}/v1/notebooks/${sessionId}/cells/${id}`, { source })
    cellIds.push(id)
  }
  return { sessionId, name, cellIds }
}

// Refused until the notebook's environment has synced.
async function execute(baseUrl, sessionId, cellId) {
  const url = `${baseUrl}/v1/notebooks/${sessionId}/cells/${cellId}/execute`
  const deadline = Date.now() + SYNC_TIMEOUT_MS
  for (;;) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
    })
    if (res.ok) return res.json()
    if (Date.now() > deadline) throw new Error(`POST ${url} -> ${res.status}: ${await res.text()}`)
    await sleep(1000)
  }
}

async function main() {
  const baseUrl = argFor('--base-url', DEFAULT_BASE_URL).replace(/\/$/, '')
  const parentPath = argFor('--parent-path', '/tmp')
  const stamp = Date.now()
  // `a` is read by the next cell, so running both stores it.
  const full = await createNotebook(baseUrl, parentPath, `store_smoke_full_${stamp}`, [
    'a = 1\n',
    'b = a + 1\n',
  ])
  const empty = await createNotebook(baseUrl, parentPath, `store_smoke_empty_${stamp}`, [])
  for (const cellId of full.cellIds) await execute(baseUrl, full.sessionId, cellId)

  const expected = {}
  for (const nb of [full, empty]) {
    const base = `${baseUrl}/v1/notebooks/${nb.sessionId}/store`
    const list = await request('GET', `${base}/artifacts?limit=100`)
    const stats = await request('GET', `${base}/stats`)
    expected[nb.sessionId] = {
      scope: nb.name,
      rows: list.artifacts.length,
      versions: stats.total_versions.toLocaleString('en-US'),
    }
  }
  if (expected[full.sessionId].rows === 0) throw new Error('running the cells stored nothing')

  const failures = []
  const browser = await chromium.launch({ headless: true })
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
    // A route change in place, as the app's own links make one.
    const go = (route) =>
      page.evaluate((r) => {
        window.location.hash = r
      }, route)
    const shown = () =>
      page.evaluate(() => ({
        scope: document.querySelector('[data-testid="artifacts-scope"]')?.textContent.trim(),
        rows: document.querySelectorAll('[data-testid^="artifact-row-"]').length,
        versions:
          document
            .querySelector('[data-testid="artifact-stats"] .stat-value')
            ?.textContent.trim() ?? null,
      }))
    const settled = () =>
      page.waitForFunction(
        () => document.querySelector('[data-testid="artifacts-refresh"]')?.disabled === false,
        null,
        { timeout: TIMEOUT_MS },
      )
    const expectStore = async (label, sessionId) => {
      const want = expected[sessionId]
      try {
        await page.waitForFunction(
          (scope) =>
            document.querySelector('[data-testid="artifacts-scope"]')?.textContent.trim() === scope,
          want.scope,
          { timeout: 5000 },
        )
        await settled()
      } catch {
        // Reported below from what the page shows.
      }
      const got = await shown()
      for (const key of ['scope', 'rows', 'versions']) {
        if (got[key] !== want[key]) {
          failures.push(
            `${label}: ${key} is ${JSON.stringify(got[key])}, not ${JSON.stringify(want[key])}`,
          )
        }
      }
    }

    await page.goto(`${baseUrl}/#/artifacts`)
    await settled()

    // 1.
    await go(`/notebook/${full.sessionId}/artifacts`)
    await expectStore('server store to a notebook', full.sessionId)

    // 2.
    await go(`/notebook/${empty.sessionId}/artifacts`)
    await expectStore('one notebook to another', empty.sessionId)

    // 3.
    await page.route(`**/v1/notebooks/${full.sessionId}/store/**`, async (route) => {
      await sleep(SLOW_MS)
      await route.continue()
    })
    await go(`/notebook/${full.sessionId}/artifacts`)
    await sleep(200)
    await go(`/notebook/${empty.sessionId}/artifacts`)
    await sleep(SLOW_MS + 1000)
    await expectStore('a slow load for the store just left', empty.sessionId)
    await page.unroute(`**/v1/notebooks/${full.sessionId}/store/**`)

    // 4.
    await go(`/notebook/${full.sessionId}/artifacts`)
    await expectStore('back to the full notebook', full.sessionId)
    await page.route(`**/v1/notebooks/${empty.sessionId}/store/**`, async (route) => {
      await sleep(SLOW_MS)
      await route.continue()
    })
    await go(`/notebook/${empty.sessionId}/artifacts`)
    await sleep(200)
    const loading = await shown()
    if (loading.rows !== 0 || loading.versions !== null) {
      failures.push(
        `while the next store loads: the last one's ${loading.rows} rows and ${loading.versions} versions still show`,
      )
    }
    await expectStore('after the slow load', empty.sessionId)
    await page.unroute(`**/v1/notebooks/${empty.sessionId}/store/**`)

    await go('/artifacts')
    await settled()
    const back = await shown()
    if (back.scope !== 'server store') {
      failures.push(`back to the server store: scope is ${JSON.stringify(back.scope)}`)
    }
  } finally {
    await browser.close()
  }

  if (failures.length) {
    console.error('STORE PAGE SMOKE: FAIL')
    for (const f of failures) console.error('  - ' + f)
    process.exit(1)
  }
  console.log('STORE PAGE SMOKE: PASS')
}

main().catch((err) => {
  console.error('STORE PAGE SMOKE: ERROR', err)
  process.exit(1)
})
