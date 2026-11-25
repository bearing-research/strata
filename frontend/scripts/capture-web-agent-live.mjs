/**
 * Capture the frames of the home page's animation: a coding agent builds a
 * notebook through the `strata` CLI while a browser on the same session updates.
 *
 * Driven by `scripts/capture_web_agent_drive.py`, which starts the server,
 * scaffolds the notebook and assembles the GIF; run it through that script.
 *
 * The agent's words are written here; everything else is real. Each step writes
 * a cell's file, runs the CLI command an agent would, and shows its actual
 * output, and each frame waits until both browsers (one per theme) show the
 * step's effect, so a broken broadcast fails the capture. The closing summary
 * quotes numbers the cells printed.
 */
import { execFile } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import process from 'node:process'
import { promisify } from 'node:util'
import { chromium } from 'playwright'

const run = promisify(execFile)

// The README and the docs show the GIF about 880px wide, near half its size, so the
// agent's pane is wide with large type and the browser no wider than it must be.
const APP = { width: 1024, height: 760 }
const PANE_WIDTH = 560
const THEMES = ['light', 'dark']
const TIMEOUT_MS = 180000

const REQUEST = "Plot CO₂ at Mauna Loa since 1958, and how fast it's rising each decade."

// The cells the agent writes. growth's first draft names a column that is not there.
const LOAD = `# @name load
# @fetch co2 https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv
import pandas as pd

raw = pd.read_csv(co2, comment="#")
monthly = raw[["year", "month", "average"]].rename(columns={"average": "co2_ppm"})
print(f"{len(monthly)} months, {monthly.year.min()} to {monthly.year.max()}")
monthly
`
const GROWTH = (column) => `# @name growth
full = monthly.groupby("year").filter(lambda y: len(y) == 12)
yearly = full.groupby("year")["${column}"].mean()
decades = yearly.diff().dropna().groupby(lambda y: f"{y // 10 * 10}s").mean().round(2)
print(decades.to_string())
decades.rename("ppm a year").to_frame()
`
const PLOT = `# @name plot
import matplotlib.pyplot as plt

plt.style.use("strata")


def draw(record):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(record.year + (record.month - 0.5) / 12, record.co2_ppm, linewidth=1)
    ax.set_title("CO₂ at Mauna Loa, ppm")
    return fig


draw(monthly)
`

function argFor(flag) {
  const i = process.argv.indexOf(flag)
  if (i < 0 || i + 1 >= process.argv.length) throw new Error(`${flag} is required`)
  return process.argv[i + 1]
}

function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

async function main() {
  const baseUrl = argFor('--base-url').replace(/\/$/, '')
  const sessionId = argFor('--session')
  const python = argFor('--python')
  const agentDir = argFor('--agent-dir')
  const outDir = argFor('--out')
  mkdirSync(outDir, { recursive: true })
  mkdirSync(agentDir, { recursive: true })

  // What the agent's pane shows, oldest first.
  const pane = [{ kind: 'user', text: REQUEST }]
  const say = (text) => pane.push({ kind: 'agent', text })

  function write(file, code) {
    writeFileSync(`${agentDir}/${file}`, code)
    pane.push({ kind: 'write', file, code })
  }

  function edit(file, before, after) {
    writeFileSync(`${agentDir}/${file}`, after)
    const was = before.split('\n')
    const now = after.split('\n')
    const changed = now.findIndex((line, i) => line !== was[i])
    pane.push({ kind: 'edit', file, removed: was[changed], added: now[changed] })
  }

  /**
   * Run `strata <args>` against the live session as the agent, in its directory;
   * the pane shows the command (without `--server`/`--session`/`--author`, the same
   * on every call) and its real output. A failing run is shown, not thrown: the
   * agent reads the error.
   */
  async function strata(...args) {
    const entry = { kind: 'run', command: `strata ${args.join(' ')}`, output: '' }
    pane.push(entry)
    const full = [...args, '--format', 'human', '--server', baseUrl, '--session', sessionId]
    try {
      const { stdout } = await run(python, ['-m', 'strata.cli', ...full, '--author', 'agent'], {
        cwd: agentDir,
      })
      entry.output = stdout.trimEnd()
    } catch (err) {
      if (err.stdout === undefined) throw err
      entry.output = err.stdout.trimEnd()
      entry.failed = true
    }
    return entry.output
  }

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
      // The mode fails closed to service until the worker catalog syncs, so it
      // doubles as the settle signal.
      await page.locator('.header[data-mode="personal"]').waitFor({
        state: 'attached',
        timeout: TIMEOUT_MS,
      })
      pages[theme] = page
    }
    // One page renders every frame: the agent's pane beside a browser screenshot.
    const composer = await (
      await browser.newContext({ viewport: { width: 800, height: 600 } })
    ).newPage()

    /**
     * Wait until *ready* holds on both pages, scroll *cell* to *block* (its code at
     * the top, or its output at the bottom), then take one frame per theme.
     */
    async function beat(what, ready, cell, block = 'start') {
      for (const theme of THEMES) {
        try {
          await ready(pages[theme])
        } catch (err) {
          const states = await pages[theme]
            .locator('.cell')
            .evaluateAll((els) =>
              els.map((el) => [...el.classList].filter((c) => c.startsWith('state-')).join(' ')),
            )
          throw new Error(
            `cells are [${states.join(' | ')}]; ` +
              `the ${theme} browser never showed ${what}: the live mirror is broken, ` +
              `or the beat needs longer (${err.message})`,
          )
        }
      }
      for (const theme of THEMES) {
        const page = pages[theme]
        if (cell !== undefined) {
          await page
            .locator('.cell')
            .nth(cell)
            .evaluate((el, where) => el.scrollIntoView({ block: where }), block)
        }
        // Status flips a moment before the output paints; let it land.
        await page.waitForTimeout(700)
        const shot = await page.screenshot()
        const path = `${outDir}/web-agent-live-${theme}-${String(frames[theme].length).padStart(2, '0')}.png`
        await composer.setContent(frameHtml(theme, shot, pane))
        await composer.locator('.frame').screenshot({ path })
        frames[theme].push(path)
      }
    }

    const cells = (page) => page.locator('.cell')
    // Cell *n* shows one of *states*. A cell another reads from reads cached once it is reused.
    const settled =
      (n, ...states) =>
      (page) =>
        page.waitForFunction(
          ([index, wanted]) => {
            const cell = document.querySelectorAll('.cell')[index]
            return (
              cell !== undefined &&
              wanted.some((state) => cell.classList.contains(`state-${state}`))
            )
          },
          [n, states],
          { timeout: TIMEOUT_MS, polling: 200 },
        )
    await beat('the empty notebook', async (page) => {
      if ((await cells(page).count()) !== 0) throw new Error('expected no cells yet')
    })

    say("I'll start with NOAA's monthly record from Mauna Loa, in a cell of its own.")
    write('load.py', LOAD)
    const added = await strata('cell', 'add', '--run', '--file', 'load.py')
    if (!added.includes('ok ')) throw new Error(`load did not run:\n${added}`)
    await beat('the load cell finishing', settled(0, 'ready', 'cached'), 0)

    say('Now the yearly means, and how much each year rises, averaged by decade.')
    write('growth.py', GROWTH('ppm'))
    const failed = await strata('cell', 'add', '--run', '--file', 'growth.py')
    const growthId = failed.split(/\s+/)[1]
    if (!failed.includes('KeyError')) throw new Error(`growth did not fail as drafted:\n${failed}`)
    await beat('the growth cell failing', settled(1, 'error'), 1)

    say(
      'The column is co2_ppm. Fixing that line; load stays as it is, so the data is not fetched again.',
    )
    edit('growth.py', GROWTH('ppm'), GROWTH('co2_ppm'))
    await strata('cell', 'edit', growthId, '--file', 'growth.py')
    const grown = await strata('cell', 'run', growthId)
    await beat('the growth cell fixed', settled(1, 'ready', 'cached'), 1)

    say('And the curve itself.')
    write('plot.py', PLOT)
    await strata('cell', 'add', '--run', '--file', 'plot.py')
    await beat('the plot', settled(2, 'ready', 'cached'), 2, 'end')

    await strata('status')
    say(summary(added, grown))
    await beat('the finished notebook', settled(2, 'ready', 'cached'), 2, 'end')
  } finally {
    await browser.close()
  }
  writeFileSync(`${outDir}/frames.json`, JSON.stringify(frames, null, 2))
  for (const theme of THEMES) console.log(`  ${frames[theme].length} ${theme} frames`)
}

/** The agent's last words, quoting what the cells printed. */
function summary(loaded, grown) {
  const span = loaded.match(/(\d+) months, (\d{4}) to (\d{4})/)
  const decades = [...grown.matchAll(/^(\d{4}s)\s+([\d.]+)$/gm)]
  if (!span || decades.length < 2) throw new Error(`unexpected output:\n${loaded}\n${grown}`)
  const [first, last] = [decades[0], decades[decades.length - 1]]
  return (
    `Done: three cells, ${span[1]} months from ${span[2]} to ${span[3]}. ` +
    `The rise sped up from ${first[2]} ppm a year in the ${first[1]} to ${last[2]} in the ${last[1]} so far. ` +
    'Fixing growth reran only growth.'
  )
}

/** A frame: the agent's pane on the left, the browser on the right. */
function frameHtml(theme, shot, pane) {
  const dark = theme === 'dark'
  const page = dark ? '#11111b' : '#dce0e8'
  const chrome = dark ? '#313244' : '#bcc0cc'
  const chromeText = dark ? '#cdd6f4' : '#4c4f69'
  const code = (text, cls = '') => `<pre class="code ${cls}">${escapeHtml(text.trimEnd())}</pre>`
  const entries = pane.map((entry) => {
    switch (entry.kind) {
      case 'user':
        return `<div class="user">${escapeHtml(entry.text)}</div>`
      case 'agent':
        return `<div class="agent">${escapeHtml(entry.text)}</div>`
      case 'write':
        return `<div class="tool"><span class="tag">write</span> ${entry.file}${code(entry.code)}</div>`
      case 'edit':
        return (
          `<div class="tool"><span class="tag">edit</span> ${entry.file}` +
          `${code(`- ${entry.removed}`, 'del')}${code(`+ ${entry.added}`, 'add')}</div>`
        )
      case 'run':
        return (
          `<div class="tool"><span class="tag">run</span> <code>${escapeHtml(entry.command)}</code>` +
          `${entry.output ? code(entry.output, entry.failed ? 'out failed' : 'out') : ''}</div>`
        )
      default:
        throw new Error(`unknown pane entry ${entry.kind}`)
    }
  })
  return `<!doctype html><html><head><style>
    body { margin: 0; background: ${page}; font-family: -apple-system, 'Segoe UI', sans-serif; }
    .frame { display: inline-flex; gap: 16px; padding: 16px; background: ${page}; }
    .win { border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,.25); }
    .bar { height: 30px; background: ${chrome}; color: ${chromeText}; font-size: 14px;
           display: flex; align-items: center; justify-content: center; position: relative; }
    .bar::before { content: ''; position: absolute; left: 12px; width: 10px; height: 10px;
           border-radius: 50%; background: #f38ba8; box-shadow: 16px 0 #f9e2af, 32px 0 #a6e3a1; }
    .pane { width: ${PANE_WIDTH}px; height: ${APP.height}px; background: #1e1e2e; color: #cdd6f4;
            padding: 16px 18px; box-sizing: border-box; overflow: hidden;
            display: flex; flex-direction: column; justify-content: flex-end; gap: 12px;
            font-size: 18px; line-height: 1.45; }
    .user { background: #313244; border-left: 3px solid #89b4fa; padding: 8px 10px;
            border-radius: 4px; color: #cdd6f4; }
    .agent { color: #cdd6f4; }
    .tool { color: #a6adc8; font-size: 16px; }
    .tool code { font: 16px 'JetBrains Mono', Menlo, monospace; color: #a6e3a1; }
    .tag { display: inline-block; font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
           color: #1e1e2e; background: #a6adc8; border-radius: 3px; padding: 0 5px; }
    .code { margin: 5px 0 0; padding: 6px 8px; background: #181825; border-radius: 4px;
            font: 15px/1.4 'JetBrains Mono', Menlo, monospace; color: #bac2de;
            white-space: pre-wrap; word-break: break-word; }
    .out { color: #cdd6f4; }
    .failed { color: #f38ba8; }
    .del { color: #f38ba8; margin-top: 4px; }
    .add { color: #a6e3a1; margin-top: 2px; }
    img { display: block; width: ${APP.width}px; height: ${APP.height}px; }
  </style></head><body><div class="frame">
    <div class="win"><div class="bar">coding agent, driving the strata CLI</div><div class="pane">${entries.join('')}</div></div>
    <div class="win"><div class="bar">browser: the same notebook</div>
      <img src="data:image/png;base64,${shot.toString('base64')}"></div>
  </div></body></html>`
}

await main()
