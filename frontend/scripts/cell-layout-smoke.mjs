/**
 * Browser smoke for how a cell lays out its actions. CSS decides all of this, and the frontend has
 * no component tests, so only a real browser can see it break.
 *
 *   1. a short cell is as tall as its contents: its actions don't hold it open
 *   2. hovering opens the actions without moving anything
 *   3. every action can be clicked, including those reaching over the cell below
 *   4. a keyboard reaches them too
 *   5. the notebook header fits a phone: nothing overlaps or runs off the page, its menus included
 *   6. so does the Logs and Artifacts pages' header, which stays one row on a desktop
 *
 * Exits non-zero on any failed assertion.
 *
 *   node scripts/cell-layout-smoke.mjs --base-url http://127.0.0.1:8770
 */
import process from 'node:process'
import { chromium } from 'playwright'

const DEFAULT_BASE_URL = 'http://127.0.0.1:8770'
const TIMEOUT_MS = 20000
// A new notebook's first environment sync.
const SYNC_TIMEOUT_MS = 180000
// The cell's own border.
const BORDER_PX = 2

function argFor(flag, fallback) {
  const i = process.argv.indexOf(flag)
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback
}

async function request(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${method} ${url} -> ${res.status}: ${await res.text()}`)
  return res.json()
}

async function addCell(baseUrl, sessionId, source) {
  const { id } = await request('POST', `${baseUrl}/v1/notebooks/${sessionId}/cells`, {
    language: 'python',
  })
  await request('PUT', `${baseUrl}/v1/notebooks/${sessionId}/cells/${id}`, { source })
}

async function main() {
  const baseUrl = argFor('--base-url', DEFAULT_BASE_URL).replace(/\/$/, '')
  const created = await request('POST', `${baseUrl}/v1/notebooks/create`, {
    parent_path: argFor('--parent-path', '/tmp'),
    name: `cell_layout_smoke_${Date.now()}`,
  })
  const sessionId = created.session_id
  await addCell(baseUrl, sessionId, 'first = 1\n')
  await addCell(baseUrl, sessionId, 'second = 2\n')

  const failures = []
  const browser = await chromium.launch({ headless: true })
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
    await page.goto(`${baseUrl}/#/notebook/${sessionId}`, { timeout: TIMEOUT_MS })
    const cell = page.locator('.cell').first()
    const actions = cell.locator('.cell-actions')
    await page.locator('.cell').nth(1).waitFor({ timeout: TIMEOUT_MS })
    // The sync disables Run, which would read as an action nobody can click.
    const run = await actions.getByRole('button', { name: 'Run', exact: true }).elementHandle()
    await page.waitForFunction((el) => !el.disabled, run, { timeout: SYNC_TIMEOUT_MS })

    const heightOf = async (locator) => (await locator.boundingBox()).height
    const opacityOf = () => actions.evaluate((el) => getComputedStyle(el).opacity)
    // Opacity fades in, so a single read could catch it partway.
    const opens = async () => {
      const handle = await actions.elementHandle()
      return page
        .waitForFunction((el) => getComputedStyle(el).opacity === '1', handle, {
          timeout: TIMEOUT_MS,
        })
        .then(() => true)
        .catch(() => false)
    }

    // 1. At rest.
    await page.mouse.move(0, 0)
    const atRest = await heightOf(cell)
    // Down to its last child: the body itself stretches to whatever holds the cell open.
    const contents = await cell.locator('.cell-body').evaluate((body) => {
      const bottoms = [...body.children].map((c) => c.getBoundingClientRect().bottom)
      return Math.max(...bottoms) - body.getBoundingClientRect().top
    })
    if (atRest - contents > BORDER_PX) {
      failures.push(`a short cell is ${atRest}px tall around ${contents}px of contents`)
    }
    if ((await opacityOf()) !== '0') failures.push('actions show on a cell nobody is pointing at')

    // 2 + 3. Under the pointer. A trial click fails if anything covers the button.
    await cell.hover()
    if ((await heightOf(cell)) !== atRest) failures.push('hovering a cell changed its height')
    const buttons = actions.locator('button')
    const count = await buttons.count()
    if (count === 0) failures.push('a cell has no action buttons')
    for (let i = 0; i < count; i++) {
      const name = await buttons.nth(i).getAttribute('aria-label')
      try {
        await buttons.nth(i).click({ trial: true, timeout: TIMEOUT_MS })
      } catch {
        failures.push(`the "${name}" action cannot be clicked`)
      }
    }

    // 4. By keyboard, with the pointer elsewhere.
    await page.mouse.move(0, 0)
    let reached = false
    for (let i = 0; i < 40 && !reached; i++) {
      await page.keyboard.press('Tab')
      reached = await actions.evaluate((el) => el.contains(document.activeElement))
    }
    if (!reached) failures.push('tabbing never reached a cell action')
    else if (!(await opens())) failures.push('a focused action stays hidden')

    // 5. At a phone's width.
    for (const width of [360, 390, 600]) {
      await page.setViewportSize({ width, height: 800 })
      const fit = await page.evaluate(() => {
        const parts = [
          ...document.querySelectorAll('header.header .logo, header.header .notebook-name'),
          ...document.querySelectorAll('header.header .header-right > *'),
        ].map((el) => el.getBoundingClientRect())
        const overlaps = parts.some((a, i) =>
          parts
            .slice(i + 1)
            .some(
              (b) =>
                a.left < b.right - 1 &&
                b.left < a.right - 1 &&
                a.top < b.bottom - 1 &&
                b.top < a.bottom - 1,
            ),
        )
        return { overlaps, scrollWidth: document.documentElement.scrollWidth }
      })
      if (fit.overlaps) failures.push(`the header's parts overlap at ${width}px`)
      if (fit.scrollWidth > width) {
        failures.push(`the page is ${fit.scrollWidth}px wide at ${width}px`)
      }
      for (const trigger of [
        'add-cell-menu-trigger',
        'export-menu-trigger',
        'notebook-overflow-trigger',
      ]) {
        await page.locator('header.header').getByTestId(trigger).click()
        const menu = await page.getByRole('menu').first().boundingBox()
        if (!menu || menu.x < 0 || menu.x + menu.width > width) {
          failures.push(`the ${trigger} menu runs off the page at ${width}px`)
        }
        await page.keyboard.press('Escape')
        await page.mouse.click(1, 790)
      }
    }

    // 6. The Logs and Artifacts pages' header: fits a phone, one row on a desktop.
    for (const route of ['/logs', '/artifacts', `/notebook/${sessionId}/artifacts`]) {
      for (const width of [360, 390, 600, 700, 1024]) {
        await page.setViewportSize({ width, height: 800 })
        await page.goto(`${baseUrl}/#${route}`)
        await page.getByTestId('page-header').waitFor({ timeout: TIMEOUT_MS })
        if (route.startsWith('/notebook/')) {
          await page
            .getByText(/cell_layout_smoke_/)
            .waitFor({ state: 'attached', timeout: TIMEOUT_MS })
        }
        const fit = await page.evaluate(() => {
          const header = document.querySelector('[data-testid="page-header"]')
          const rects = [
            ...header.querySelectorAll('.back-btn, .home-link, .crumbs > *, .page-actions > *'),
          ].map((el) => el.getBoundingClientRect())
          const overlaps = rects.some((a, i) =>
            rects
              .slice(i + 1)
              .some(
                (b) =>
                  a.left < b.right - 1 &&
                  b.left < a.right - 1 &&
                  a.top < b.bottom - 1 &&
                  b.top < a.bottom - 1,
              ),
          )
          return {
            overlaps,
            offPage: rects.some((r) => r.left < 0 || r.right > innerWidth),
            // "strata" under its diamond, a two-line page name.
            broken: [...header.querySelectorAll('.home-link, .nav-link, .page-name')].some(
              (el) => el.getClientRects().length > 1,
            ),
            squeezed: [...header.querySelectorAll('.page-name')].some(
              (el) => el.getBoundingClientRect().width < parseFloat(getComputedStyle(el).fontSize),
            ),
            // Only the notebook's name may end in an ellipsis.
            clipped: [...header.querySelectorAll('.page-name:last-child')].some(
              (el) => el.scrollWidth > el.clientWidth,
            ),
            // The links share Back's row.
            linksBelow:
              header.querySelector('.page-actions').getBoundingClientRect().top >=
              header.querySelector('.back-btn').getBoundingClientRect().bottom,
            height: header.getBoundingClientRect().height,
            scrollWidth: document.documentElement.scrollWidth,
          }
        })
        const where = `${route} at ${width}px`
        if (fit.overlaps) failures.push(`the page header's parts overlap on ${where}`)
        if (fit.offPage) failures.push(`the page header runs off the page on ${where}`)
        if (fit.broken) failures.push(`a page header link or name breaks across lines on ${where}`)
        if (fit.squeezed) failures.push(`a page name is squeezed out on ${where}`)
        if (fit.clipped) failures.push(`the page's name is cut short on ${where}`)
        if (fit.linksBelow) failures.push(`the page header's links drop below Back on ${where}`)
        if (fit.scrollWidth > width) failures.push(`${where} is ${fit.scrollWidth}px wide`)
        if (width > 600 && fit.height > 40) {
          failures.push(`the page header is ${Math.round(fit.height)}px tall on ${where}`)
        }
      }
    }
  } finally {
    await browser.close()
  }

  if (failures.length) {
    console.error('CELL LAYOUT SMOKE: FAIL')
    for (const f of failures) console.error('  - ' + f)
    process.exit(1)
  }
  console.log('CELL LAYOUT SMOKE: PASS')
}

main().catch((err) => {
  console.error('CELL LAYOUT SMOKE: ERROR', err)
  process.exit(1)
})
