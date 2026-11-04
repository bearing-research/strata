/**
 * Capture the frames of the cache-payoff animation, from the tutorial notebook: edit one cell,
 * only the cells that read it run again, and the cell above keeps what it saved.
 *
 * Driven by `scripts/capture_cache_payoff.py`, which starts the server, opens the tutorial and
 * assembles the GIF; run it through that script.
 *
 * Nothing is canned. Each beat clicks and types in a real browser and waits until the cells reach
 * the states it names, so a cache that stopped working fails the capture rather than producing a
 * convincing GIF. Both themes watch one session: light is driven, dark follows.
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import process from 'node:process'
import { chromium } from 'playwright'

const APP = { width: 1280, height: 1100 }
const THEMES = ['light', 'dark']
const TIMEOUT_MS = 120000

function argFor(flag) {
  const i = process.argv.indexOf(flag)
  if (i < 0 || i + 1 >= process.argv.length) throw new Error(`${flag} is required`)
  return process.argv[i + 1]
}

async function main() {
  const baseUrl = argFor('--base-url').replace(/\/$/, '')
  const sessionId = argFor('--session')
  const outDir = argFor('--out')
  mkdirSync(outDir, { recursive: true })

  const browser = await chromium.launch()
  const frames = { light: [], dark: [] }
  try {
    const pages = {}
    for (const theme of THEMES) {
      const context = await browser.newContext({ viewport: APP, deviceScaleFactor: 1 })
      await context.addInitScript((value) => {
        try {
          localStorage.setItem('strata.theme', value)
          localStorage.setItem('strata:dagDrawerCollapsed', '0')
        } catch {
          // Storage disabled: the page falls back to the system preference.
        }
      }, theme)
      const page = await context.newPage()
      await page.goto(`${baseUrl}/#/notebook/${sessionId}`, { waitUntil: 'networkidle' })
      await page.locator('.header[data-mode="personal"]').waitFor({
        state: 'attached',
        timeout: TIMEOUT_MS,
      })
      // Collapsed with the cell's own control, so every cell's badge fits one frame. The edited
      // cell stays open: its table is what changes.
      const toCollapse = page.locator('.cell').filter({
        hasNot: page.getByText('@name Summarize by region'),
      })
      for (let i = 0; i < (await toCollapse.count()); i++) {
        await toCollapse.nth(i).hover()
        await toCollapse.nth(i).getByRole('button', { name: 'Collapse cell' }).click()
      }
      pages[theme] = page
    }
    const composer = await (
      await browser.newContext({ viewport: { width: 800, height: 600 } })
    ).newPage()

    const cell = (page, name) => page.locator('.cell').filter({ hasText: `@name ${name}` })
    const inState = (page, name, state) =>
      cell(page, name)
        .and(page.locator(`.state-${state}`))
        .waitFor({ timeout: TIMEOUT_MS })

    /** Wait until the cells are in these states on both pages, then take one frame per theme. */
    async function beat(caption, states) {
      for (const theme of THEMES) {
        for (const [name, state] of Object.entries(states)) {
          try {
            await inState(pages[theme], name, state)
          } catch (err) {
            throw new Error(`${theme}: "${name}" never read ${state} (${err.message})`)
          }
        }
      }
      for (const theme of THEMES) {
        await pages[theme].mouse.move(0, 0)
        await pages[theme]
          .locator('[data-testid="notebook-cells-panel"]')
          .evaluate((panel) => panel.scrollTo(0, 0))
        await pages[theme].waitForTimeout(700)
        const shot = await pages[theme].screenshot()
        const n = String(frames[theme].length).padStart(2, '0')
        const path = `${outDir}/cache-payoff-${theme}-${n}.png`
        await composer.setContent(frameHtml(theme, shot, caption))
        await composer.locator('.frame').screenshot({ path })
        frames[theme].push(path)
      }
    }

    const driver = pages.light
    const runAll = () => driver.getByRole('button', { name: 'Run All' }).first().click()
    const LOAD = 'Load the data'
    const SUMMARIZE = 'Summarize by region'
    const PLOT = 'Plot it'

    await beat('The tutorial notebook. Nothing has run.', {
      [LOAD]: 'idle',
      [SUMMARIZE]: 'idle',
      [PLOT]: 'idle',
    })

    await runAll()
    await beat('Run All: every cell computes.', {
      [LOAD]: 'ready',
      [SUMMARIZE]: 'ready',
      [PLOT]: 'ready',
    })

    await runAll()
    await beat('Run All again: nothing runs. Every code cell is a cache hit.', {
      [LOAD]: 'cached',
      [SUMMARIZE]: 'cached',
      [PLOT]: 'cached',
    })

    await cell(driver, SUMMARIZE).locator('.cm-line', { hasText: '.sum()' }).click()
    await driver.keyboard.press('End')
    for (let i = 0; i < '.sum()'.length; i++) await driver.keyboard.press('Backspace')
    await driver.keyboard.type('.mean()')
    // Leaving the editor sends the edit.
    await driver
      .locator('.header')
      .first()
      .click({ position: { x: 300, y: 10 } })
    await beat('Edit one cell: it and the chart below read stale. The load cell stays cached.', {
      [LOAD]: 'cached',
      [SUMMARIZE]: 'stale',
      [PLOT]: 'stale',
    })

    await runAll()
    await beat('Run All: only those two run. The load cell keeps what it saved.', {
      [LOAD]: 'cached',
      [SUMMARIZE]: 'ready',
      [PLOT]: 'ready',
    })
  } finally {
    await browser.close()
  }
  writeFileSync(`${outDir}/frames.json`, JSON.stringify(frames, null, 2))
  for (const theme of THEMES) console.log(`  ${frames[theme].length} ${theme} frames`)
}

/** A frame: the browser under a line saying what just happened, and that it is real. */
function frameHtml(theme, shot, caption) {
  const dark = theme === 'dark'
  const page = dark ? '#11111b' : '#dce0e8'
  const text = dark ? '#cdd6f4' : '#4c4f69'
  const muted = dark ? '#a6adc8' : '#5c5f77'
  return `<!doctype html><html><head><style>
    body { margin: 0; background: ${page}; font-family: -apple-system, 'Segoe UI', sans-serif; }
    .frame { display: inline-block; padding: 12px 16px 16px; background: ${page}; }
    .line { display: flex; justify-content: space-between; align-items: baseline; gap: 24px;
            padding-bottom: 10px; color: ${text}; font-size: 16px; font-weight: 600; }
    .real { color: ${muted}; font-size: 12px; font-weight: 400; white-space: nowrap; }
    img { display: block; width: ${APP.width}px; height: ${APP.height}px; border-radius: 8px; }
  </style></head><body><div class="frame">
    <div class="line"><span>${caption}</span><span class="real">Real capture of strata tutorial</span></div>
    <img src="data:image/png;base64,${shot.toString('base64')}">
  </div></body></html>`
}

await main()
