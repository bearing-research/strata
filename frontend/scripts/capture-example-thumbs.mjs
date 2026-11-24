/**
 * Capture one screenshot per example notebook, for the gallery's cards.
 *
 * Driven by `scripts/capture_docs_shots.py --only examples`, which runs each
 * example and serves the copies; this script only drives the browser:
 *
 *   uv run python scripts/capture_docs_shots.py --only examples
 *
 * `--manifest` is a JSON list of `{name, path, focus, run}`; `run` executes the
 * notebook's cells on the server first. `focus` is text in the
 * cell to frame; without it the shot centres the first plot, else table, else
 * rendered markdown, else output of any kind. Writes `<name>-<theme>.png` to
 * `--out`; the Python side downsizes them to the WebP the docs embed.
 */
import { readFileSync } from 'node:fs'
import { chromium } from 'playwright'
import { api, argFor, loadNotebook, openNotebook, THEMES, TIMEOUT_MS } from './docs-shots-lib.mjs'

// 16:10, narrow enough that a cell's output fills the frame rather than a margin.
const VIEWPORT = { width: 1200, height: 750 }

// Most telling first: a plot says more about an example than a table does.
const OUTPUT_PREFERENCE = [
  '.output-image',
  '.cell-output table',
  '.output-markdown',
  '.cell-output',
]
// How much of the cell's source shows above its output.
const SOURCE_LEAD_PX = 200

async function frame(page, focus) {
  const cells = page.locator('.cell')
  const cell = focus ? cells.filter({ hasText: focus }).first() : null
  if (cell) await cell.waitFor({ timeout: TIMEOUT_MS })
  // A plot's height is unknown until it decodes; scrolling before that frames
  // the output lower than it ends up.
  await page.waitForFunction(() => [...document.images].every((img) => img.complete), null, {
    timeout: TIMEOUT_MS,
  })
  for (const selector of OUTPUT_PREFERENCE) {
    const output = (cell ?? page).locator(selector).first()
    if ((await output.count()) > 0) {
      // Scroll the whole output block, not the element: centring a table
      // would also scroll the grid's own scroller past its first rows.
      await output.evaluate((el, lead) => {
        const block = el.closest('.cell-output') ?? el
        block.scrollIntoView({ block: 'start' })
        // Keep the end of the cell's source above the output in frame.
        let scroller = block.parentElement
        const scrolls = (node) =>
          /auto|scroll/.test(getComputedStyle(node).overflowY) &&
          node.scrollHeight > node.clientHeight
        while (scroller && !scrolls(scroller)) {
          scroller = scroller.parentElement
        }
        ;(scroller ?? document.scrollingElement).scrollBy(0, -lead)
      }, SOURCE_LEAD_PX)
      return
    }
  }
  // Nothing rendered: frame the cell's source instead of an arbitrary scroll.
  await (cell ?? cells.first()).evaluate((el) => el.scrollIntoView({ block: 'start' }))
}

async function main() {
  const baseUrl = argFor('--base-url', 'http://127.0.0.1:8765').replace(/\/$/, '')
  const outDir = argFor('--out')
  const manifestPath = argFor('--manifest')
  if (!outDir || !manifestPath) throw new Error('--out and --manifest are required')
  const examples = JSON.parse(readFileSync(manifestPath, 'utf-8'))

  const browser = await chromium.launch()
  try {
    for (const example of examples) {
      const { sessionId, cells } = await openNotebook(baseUrl, example.path)
      if (example.run) {
        // Examples whose cells need the server's settings (an LLM key) run here,
        // in order; each request returns once its cell has.
        for (const cell of cells) {
          const url = `${baseUrl}/v1/notebooks/${sessionId}/cells/${cell.id}/execute`
          const result = await api('POST', url, {})
          // A card must never show an error.
          if (result.status !== 'ready') {
            throw new Error(
              `${example.name}: cell ${cell.id} ended ${result.status}: ${result.error}`,
            )
          }
        }
      }
      for (const theme of THEMES) {
        const { context, page } = await loadNotebook(browser, baseUrl, sessionId, theme, VIEWPORT, {
          drawer: false,
          scale: 1,
        })
        await frame(page, example.focus)
        // Let images decode and the grid lay out after the scroll.
        await page.waitForTimeout(600)
        await page.screenshot({ path: `${outDir}/${example.name}-${theme}.png` })
        console.log(`  ${example.name}-${theme}.png`)
        await context.close()
      }
    }
  } finally {
    await browser.close()
  }
}

await main()
