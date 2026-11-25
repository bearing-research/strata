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
 * cell to take; without it the shot takes the first plot, else table, else
 * rendered markdown, else output of any kind. Writes `<name>-<theme>.png` to
 * `--out`: that output alone, not the editor around it, which the Python side
 * sets in the card's frame and writes as the WebP the docs embed. A notebook with
 * no output at all is shot as its focus cell, or its first cell: the source alone.
 */
import { readFileSync } from 'node:fs'
import { chromium } from 'playwright'
import { api, argFor, loadNotebook, openNotebook, THEMES, TIMEOUT_MS } from './docs-shots-lib.mjs'

// Narrow enough that a cell's output fills the frame rather than a margin, and tall
// enough that a square plot fits between the editor's header and its bottom bar.
const VIEWPORT = { width: 1200, height: 1000 }

// Most telling first: a plot says more about an example than a table does.
// Each is the element the card shows.
const OUTPUT_PREFERENCE = [
  '.output-image img',
  '.cell-output .data-table',
  '.output-markdown',
  '.cell-output',
]
// A taller output keeps its top. A plot is whole up to square (a confusion matrix,
// a pair plot); a table or text is cut sooner, where its first rows say enough.
const MAX_HEIGHT_PER_WIDTH = { plot: 1, other: 0.62 }
// How much of the cell's source shows above its output.
const SOURCE_LEAD_PX = 200

async function frame(page, focus) {
  const cells = page.locator('.cell')
  let cell = null
  if (focus) {
    const match = cells.filter({ hasText: focus }).first()
    await match.waitFor({ timeout: TIMEOUT_MS })
    // By id from here: the editor draws only the lines in view, so once scrolled the
    // focus text can leave the page and a text filter would lose the cell.
    cell = page.locator(`.cell[data-cell-id="${await match.getAttribute('data-cell-id')}"]`)
  }
  // A plot's height is unknown until it decodes; scrolling before that frames
  // the output lower than it ends up.
  await page.waitForFunction(() => [...document.images].every((img) => img.complete), null, {
    timeout: TIMEOUT_MS,
  })
  for (const selector of OUTPUT_PREFERENCE) {
    const output = (cell ?? page).locator(selector).first()
    if ((await output.count()) > 0) {
      // A long markdown output starts folded behind its toggle; open it.
      if (await output.evaluate((el) => el.classList.contains('collapsed'))) {
        await output
          .locator('xpath=following-sibling::button[contains(@class, "output-toggle")]')
          .click()
      }
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
      return output
    }
  }
  // Nothing rendered: the cell itself, so the card shows its source.
  return cell ?? cells.first()
}

async function shoot(page, output, path) {
  await output.evaluate((el) => {
    // The header and bottom bar float over the page; the card shows only the output.
    // A table's own sticky header row stays.
    for (const node of document.querySelectorAll('body *')) {
      if (el.contains(node) || node.contains(el)) continue
      if (/fixed|sticky/.test(getComputedStyle(node).position)) node.style.visibility = 'hidden'
    }
    // Top first, so a tall output keeps its top.
    el.scrollIntoView({ block: 'start' })
  })
  await page.waitForTimeout(200)
  const box = await output.boundingBox()
  if (!box) throw new Error(`${path}: the element to shoot has no box`)
  const isPlot = await output.evaluate((el) => el.tagName === 'IMG')
  const ratio = isPlot ? MAX_HEIGHT_PER_WIDTH.plot : MAX_HEIGHT_PER_WIDTH.other
  // No lower than the scrolling pane's edge: below it is the editor's bottom bar.
  const paneBottom = await output.evaluate((el) => {
    let node = el.parentElement
    while (node && !/auto|scroll/.test(getComputedStyle(node).overflowY)) node = node.parentElement
    return node ? node.getBoundingClientRect().bottom : window.innerHeight
  })
  const height = Math.min(box.height, box.width * ratio, paneBottom - box.y)
  const clip = { ...box, height }
  // A plot is a transparent PNG: keep it so, and the card's frame shows through.
  await page.screenshot({ path, clip, omitBackground: true })
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
          scale: 2,
        })
        const output = await frame(page, example.focus)
        // Let images decode and the grid lay out after the scroll.
        await page.waitForTimeout(600)
        await shoot(page, output, `${outDir}/${example.name}-${theme}.png`)
        console.log(`  ${example.name}-${theme}.png`)
        await context.close()
      }
    }
  } finally {
    await browser.close()
  }
}

await main()
