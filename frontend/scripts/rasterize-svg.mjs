/**
 * Rasterise SVG files to PNG.
 *
 * The TUI screenshots Textual produces are SVG, and assembling an animation
 * needs raster frames. Nothing in the Python environment renders SVG (no
 * cairosvg, no rsvg-convert, no ImageMagick), but Playwright is already here
 * for the web captures and renders SVG exactly as a browser would — which is
 * also how a reader will eventually see it.
 *
 *   node scripts/rasterize-svg.mjs --in <dir> [--scale 1]
 *
 * Writes <name>.png beside each <name>.svg.
 */
import { readdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import process from 'node:process'
import { chromium } from 'playwright'

function argFor(flag, fallback) {
  const i = process.argv.indexOf(flag)
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback
}

const dir = resolve(argFor('--in'))
const scale = Number(argFor('--scale', '1'))
const svgs = readdirSync(dir)
  .filter((f) => f.endsWith('.svg'))
  .sort()

if (svgs.length === 0) {
  throw new Error(`no .svg files in ${dir}`)
}

const browser = await chromium.launch()
try {
  const context = await browser.newContext({ deviceScaleFactor: scale })
  const page = await context.newPage()
  for (const name of svgs) {
    const svg = readFileSync(join(dir, name), 'utf8')
    // Inline rather than file://, so the page has no chance to resolve
    // anything external — these frames must render identically offline.
    await page.setContent(
      `<!doctype html><style>html,body{margin:0;padding:0;background:transparent}
       svg{display:block}</style>${svg}`,
      { waitUntil: 'load' },
    )
    const element = await page.$('svg')
    if (!element) throw new Error(`${name}: no <svg> root after load`)
    const out = join(dir, name.replace(/\.svg$/, '.png'))
    await element.screenshot({ path: out, omitBackground: true })
    console.log(`  ${name} → ${out.split('/').pop()}`)
  }
} finally {
  await browser.close()
}
