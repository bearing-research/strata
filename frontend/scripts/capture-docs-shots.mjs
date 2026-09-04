/**
 * Capture the PNG screenshots the docs and README embed.
 *
 * Driven by `scripts/capture_docs_shots.py`, which builds the fixture
 * notebooks and serves them; this script only drives the browser. Run it
 * through that script rather than directly:
 *
 *   uv run python scripts/capture_docs_shots.py --only web
 *
 * Every shot is captured twice, once per app theme, and written as
 * `<name>-light.png` / `<name>-dark.png`. Material for MkDocs picks the
 * matching one via the `#only-light` / `#only-dark` URL suffix, so a page
 * never shows a dark screenshot on a light background.
 *
 * Waits are on *conditions*, never on timers: the mode badge fails closed to
 * "Service mode" until the worker catalog syncs, so a shot taken on a sleep
 * can label a personal-mode server as a service deployment.
 */
import process from 'node:process'
import { chromium } from 'playwright'

const VIEWPORT = { width: 1440, height: 900 }
const SCALE = 2
const THEMES = ['light', 'dark']
const TIMEOUT_MS = 30000

function argFor(flag, fallback) {
  const i = process.argv.indexOf(flag)
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback
}

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { 'content-type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${method} ${url} -> ${res.status}: ${await res.text()}`)
  return res.status === 204 ? null : res.json()
}

async function openNotebook(baseUrl, path) {
  const data = await api('POST', `${baseUrl}/v1/notebooks/open`, { path })
  return { sessionId: data.session_id, cells: data.cells }
}

/** Load a notebook page with the theme pinned, settled enough to photograph. */
async function loadNotebook(browser, baseUrl, sessionId, theme) {
  const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: SCALE })
  await context.addInitScript((value) => {
    try {
      localStorage.setItem('strata.theme', value)
    } catch {
      // Storage disabled — the page falls back to the system preference.
    }
  }, theme)
  const page = await context.newPage()

  // The header's deployment-mode badge fails closed: if GET
  // /v1/notebooks/{id}/workers fails — a 429 from the client rate limiter is
  // the easy way to get one — the UI labels a personal server "Service mode"
  // and never re-syncs. Waiting for the right label is both the settle signal
  // and the guard against photographing that wrong state; a reload is enough
  // to recover, since the next request is not rate limited.
  const badge = page.locator('.mode-badge', { hasText: 'Personal mode' })
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
      await badge.waitFor({ timeout: TIMEOUT_MS })
      lastError = null
      break
    } catch (err) {
      lastError = err
    }
  }
  if (lastError) {
    throw new Error(
      `notebook ${sessionId} never reported personal mode (a failed /workers ` +
        `request leaves the badge reading "Service mode"): ${lastError.message}`,
    )
  }
  await page.locator('.cell').first().waitFor({ timeout: TIMEOUT_MS })
  // One animation frame past the last mutation: charts and the DAG settle here.
  await page.waitForTimeout(600)
  return { context, page }
}

async function shoot(page, name, outDir, theme, target, maxWidth) {
  const path = `${outDir}/${name}-${theme}.png`
  // `target` is a CSS selector or an already-built Locator; omit it for the page.
  const locator = typeof target === 'string' ? page.locator(target) : target
  if (locator && maxWidth) {
    // The drawer spans the window but its content hugs the left edge, so an
    // element screenshot would be mostly empty. Clip it instead.
    const box = await locator.boundingBox()
    if (!box) throw new Error(`${target} has no bounding box — hidden or zero-size?`)
    await page.screenshot({
      path,
      clip: { ...box, width: Math.min(box.width, maxWidth) },
    })
  } else {
    await (locator ?? page).screenshot({ path })
  }
  console.log(`  ${name}-${theme}.png`)
}

async function main() {
  const baseUrl = argFor('--base-url', 'http://127.0.0.1:8765').replace(/\/$/, '')
  const outDir = argFor('--out', '../docs/assets')
  const irisPath = argFor('--iris-path')
  const registryPath = argFor('--registry-path')
  if (!irisPath || !registryPath) {
    throw new Error('--iris-path and --registry-path are required')
  }

  const iris = await openNotebook(baseUrl, irisPath)
  const registry = await openNotebook(baseUrl, registryPath)

  // Run the iris cells twice in this session: once with the target's cache
  // bypassed, then normally. Cache savings are priced against the last uncached
  // run *of the same cell in this session*, so without the first pass the
  // profiling panel reports "~0ms (3 hits)" — technically correct (this session
  // has no evidence of what the work costs) and a poor advertisement.
  for (const mode of ['rerun', 'normal']) {
    for (const cell of iris.cells) {
      const url = `${baseUrl}/v1/notebooks/${iris.sessionId}/cells/${cell.id}/execute?mode=${mode}`
      await api('POST', url, {})
    }
  }

  // The registry cell publishes through the ambient `strata` client, so it can
  // only run against a live server — which is why it isn't pre-run on disk.
  for (const cell of registry.cells) {
    await api('POST', `${baseUrl}/v1/notebooks/${registry.sessionId}/cells/${cell.id}/execute`, {})
  }

  // Point taxi/tip-model at champion so the Registry tab has an alias chip and
  // an audit entry. Doing it over REST rather than by clicking the menu keeps
  // the capture deterministic; the UI drives the identical route.
  const published = await api('GET', `${baseUrl}/v1/names/taxi/tip-model`)
  const artifactId = published.artifact_uri.split('/').pop().split('@')[0]
  await api('PUT', `${baseUrl}/v1/names/taxi/tip-model/aliases/champion`, {
    artifact_id: artifactId,
    version: published.version,
  })
  await api('PUT', `${baseUrl}/v1/artifacts/${artifactId}/v/${published.version}/tags`, {
    key: 'rmse',
    value: '0.19',
  })

  const browser = await chromium.launch()
  try {
    for (const theme of THEMES) {
      // 1 — the quickstart notebook, settled.
      {
        const { context, page } = await loadNotebook(browser, baseUrl, iris.sessionId, theme)
        await shoot(page, 'notebook-anatomy', outDir, theme)
        await context.close()
      }

      // 2 — the same notebook after its loader is edited: the loader falls back
      // to idle and everything downstream goes stale.
      //
      // The edit has to happen before the page loads, not while it is open: an
      // out-of-band source change (REST, CLI, MCP — anything that isn't this
      // browser tab) updates the DAG and the staleness pills but never reaches
      // the open editor, so a shot taken on a live page shows stale cells above
      // unchanged-looking source.
      {
        const loader = iris.cells[0]
        await api('PUT', `${baseUrl}/v1/notebooks/${iris.sessionId}/cells/${loader.id}`, {
          source: loader.source.replace('time.sleep(2)', 'time.sleep(1)'),
        })
        const { context, page } = await loadNotebook(browser, baseUrl, iris.sessionId, theme)
        await page.locator('text=Why stale?').first().waitFor({ timeout: TIMEOUT_MS })
        await page.waitForTimeout(600)
        await shoot(page, 'cascade-stale', outDir, theme)
        await context.close()
        await api('PUT', `${baseUrl}/v1/notebooks/${iris.sessionId}/cells/${loader.id}`, {
          source: loader.source,
        })
      }

      // 3 + 4 + 5 — the registry surfaces.
      {
        const { context, page } = await loadNotebook(browser, baseUrl, registry.sessionId, theme)
        // Every cell in the chain publishes a name, so name the one step 3 is
        // about rather than relying on cell order.
        const modelCell = page
          .locator('.cell')
          .filter({ has: page.locator('.cell-artifact-strip', { hasText: 'taxi/tip-model' }) })
        await shoot(page, 'registry-promote-strip', outDir, theme, modelCell)

        await page.locator('.drawer-tab', { hasText: 'Registry' }).click()
        await page.locator('text=taxi/tip-model').first().waitFor({ timeout: TIMEOUT_MS })
        // Expand the audit timeline: it is the third thing the page describes,
        // and collapsed it reads as an empty panel.
        // Rendered uppercase by CSS; the DOM text is "Audit (n)".
        await page.locator('.audit .toggle').click()
        await page.waitForTimeout(600)
        await shoot(page, 'registry-tab', outDir, theme, '.dag-drawer', 1040)

        // 5 — the lineage view, opened from the model's row so the chain has
        // every link the page describes rather than just the scan's root.
        await page
          .locator('tr')
          .filter({ hasText: 'taxi/tip-model' })
          .locator('.lineage-btn')
          .click()
        await page.locator('.lineage-modal').waitFor({ timeout: TIMEOUT_MS })
        await page.locator('.lineage-row').first().waitFor({ timeout: TIMEOUT_MS })
        await page.waitForTimeout(600)
        await shoot(page, 'registry-lineage', outDir, theme, '.lineage-modal')

        await context.close()
      }
    }
  } finally {
    await browser.close()
  }
}

await main()
