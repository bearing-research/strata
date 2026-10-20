/**
 * Capture the frames of the README's live web animation: the `strata` CLI
 * drives a notebook the way a coding agent does, and a real browser on the
 * same session updates as it goes.
 *
 * Driven by `scripts/capture_web_agent_drive.py`, which starts the server,
 * scaffolds the notebook and assembles the GIF; run it through that script.
 *
 * Nothing is canned. Each beat runs a real CLI command against the live
 * session and waits until both browsers (one per theme) show its effect, so a
 * broken broadcast fails the capture rather than producing a convincing GIF.
 * Each frame puts the terminal the commands ran in beside the browser.
 */
import { execFile } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import process from 'node:process'
import { promisify } from 'node:util'
import { chromium } from 'playwright'

const run = promisify(execFile)

const APP = { width: 1280, height: 880 }
const TERM_WIDTH = 420
const THEMES = ['light', 'dark']
const TIMEOUT_MS = 60000

// What the "agent" writes, through the same CLI a coding agent uses.
const LOAD = `# @name load
import time
time.sleep(2)  # stands in for a slow query
readings = [12.1, 12.4, 13.0, 12.8, 13.5]
print(f"loaded {len(readings)} readings")`
const SUMMARIZE = `# @name summarize
mean = sum(readings) / len(readings)
f"mean {mean:.2f} over {len(readings)} readings"`

function argFor(flag) {
  const i = process.argv.indexOf(flag)
  if (i < 0 || i + 1 >= process.argv.length) throw new Error(`${flag} is required`)
  return process.argv[i + 1]
}

function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

/** The command as a shell shows it: a multi-line argument continues on `>` lines. */
function promptLines(args) {
  const shown = args.map((a) => (/[\s"$]/.test(a) ? `'${a}'` : a)).join(' ')
  return shown.split('\n').map((line, i) => (i === 0 ? `$ strata ${line}` : `> ${line}`))
}

async function main() {
  const baseUrl = argFor('--base-url').replace(/\/$/, '')
  const sessionId = argFor('--session')
  const python = argFor('--python')
  const repoRoot = argFor('--repo-root')
  const outDir = argFor('--out')
  mkdirSync(outDir, { recursive: true })

  const transcript = []
  /**
   * Run `strata <args>` against the live session; its real output joins the
   * transcript. The frames leave out `--server` and `--session`, the same on
   * every line; the rest is shown as run. Human output, since the default JSON
   * would fill the pane.
   */
  function strata(command) {
    const args = [...command, '--format', 'human']
    transcript.push(...promptLines(args))
    const child = run(
      python,
      ['-m', 'strata.cli', ...args, '--server', baseUrl, '--session', sessionId],
      { cwd: repoRoot },
    )
    return child.then(({ stdout }) => {
      transcript.push(...stdout.trimEnd().split('\n'))
      return stdout
    })
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
        } catch {
          // Storage disabled: the page falls back to the system preference.
        }
      }, theme)
      const page = await context.newPage()
      await page.goto(`${baseUrl}/#/notebook/${sessionId}`, { waitUntil: 'networkidle' })
      // The mode badge fails closed to "Service mode" until the worker catalog
      // syncs, so it doubles as the settle signal.
      await page.locator('.mode-badge', { hasText: 'Personal mode' }).waitFor({
        timeout: TIMEOUT_MS,
      })
      pages[theme] = page
    }
    // One page renders every frame: the terminal beside a browser screenshot.
    const composer = await (
      await browser.newContext({ viewport: { width: 800, height: 600 } })
    ).newPage()

    /** Wait until *ready* holds on both pages, then take one frame per theme. */
    async function beat(what, ready) {
      for (const theme of THEMES) {
        try {
          await ready(pages[theme])
        } catch (err) {
          throw new Error(
            `the ${theme} browser never showed ${what}: the live mirror is broken, ` +
              `or the beat needs longer (${err.message})`,
          )
        }
      }
      for (const theme of THEMES) {
        // Status flips a moment before the output paints; let it land.
        await pages[theme].waitForTimeout(500)
        const shot = await pages[theme].screenshot()
        const path = `${outDir}/web-agent-live-${theme}-${String(frames[theme].length).padStart(2, '0')}.png`
        await composer.setContent(frameHtml(theme, shot, transcript))
        await composer.locator('.frame').screenshot({ path })
        frames[theme].push(path)
      }
    }

    const cells = (page) => page.locator('.cell')
    await beat('the empty notebook', async (page) => {
      if ((await cells(page).count()) !== 0) throw new Error('expected no cells yet')
    })

    await strata(['cell', 'add', '-c', LOAD])
    await beat('the first cell arriving', (page) =>
      cells(page).first().waitFor({ timeout: TIMEOUT_MS }),
    )
    const loadId = await pages.light.locator('.cell').first().getAttribute('data-cell-id')

    // The run takes two seconds: catch it in flight, then finished.
    const loadRun = strata(['cell', 'run', loadId])
    await beat('the first cell running', (page) =>
      page.locator('.cell.status-running').first().waitFor({ timeout: TIMEOUT_MS }),
    )
    await loadRun
    await beat('the first cell finishing', (page) =>
      page.locator('.cell.status-ready').first().waitFor({ timeout: TIMEOUT_MS }),
    )

    await strata(['cell', 'add', '-c', SUMMARIZE])
    await beat('the second cell arriving', (page) =>
      cells(page).nth(1).waitFor({ timeout: TIMEOUT_MS }),
    )
    const summarizeId = await pages.light.locator('.cell').nth(1).getAttribute('data-cell-id')

    await strata(['cell', 'run', summarizeId])
    await beat('both cells finishing', (page) =>
      page.locator('.cell.status-ready').nth(1).waitFor({ timeout: TIMEOUT_MS }),
    )

    // The payoff: running the slow cell again is a cache hit.
    await strata(['cell', 'run', loadId])
    await beat('the first cell served from the cache', (page) =>
      cells(page).first().locator('.cache-badge').waitFor({ timeout: TIMEOUT_MS }),
    )
  } finally {
    await browser.close()
  }
  writeFileSync(`${outDir}/frames.json`, JSON.stringify(frames, null, 2))
  for (const theme of THEMES) console.log(`  ${frames[theme].length} ${theme} frames`)
}

/** A frame: the agent's terminal on the left, the browser on the right. */
function frameHtml(theme, shot, transcript) {
  const dark = theme === 'dark'
  const page = dark ? '#11111b' : '#dce0e8'
  const chrome = dark ? '#313244' : '#bcc0cc'
  const chromeText = dark ? '#cdd6f4' : '#4c4f69'
  // Keep the newest lines: the terminal scrolls like a real one.
  const lines = transcript.slice(-34).map((line) => {
    const cls = line.startsWith('$ ') || line.startsWith('> ') ? 'cmd' : 'out'
    return `<div class="${cls}">${escapeHtml(line) || '&nbsp;'}</div>`
  })
  return `<!doctype html><html><head><style>
    body { margin: 0; background: ${page}; font-family: -apple-system, 'Segoe UI', sans-serif; }
    .frame { display: inline-flex; gap: 16px; padding: 16px; background: ${page}; }
    .win { border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,.25); }
    .bar { height: 28px; background: ${chrome}; color: ${chromeText}; font-size: 12px;
           display: flex; align-items: center; justify-content: center; position: relative; }
    .bar::before { content: ''; position: absolute; left: 12px; width: 10px; height: 10px;
           border-radius: 50%; background: #f38ba8; box-shadow: 16px 0 #f9e2af, 32px 0 #a6e3a1; }
    .term { width: ${TERM_WIDTH}px; height: ${APP.height}px; background: #1e1e2e; color: #cdd6f4;
            font: 12px/1.5 'JetBrains Mono', Menlo, monospace; padding: 10px 12px;
            box-sizing: border-box; white-space: pre-wrap; word-break: break-all;
            display: flex; flex-direction: column; justify-content: flex-end; }
    .cmd { color: #a6e3a1; }
    .out { color: #bac2de; }
    img { display: block; width: ${APP.width}px; height: ${APP.height}px; }
  </style></head><body><div class="frame">
    <div class="win"><div class="bar">agent: strata CLI on the live session</div><div class="term">${lines.join('')}</div></div>
    <div class="win"><div class="bar">browser: the same notebook</div>
      <img src="data:image/png;base64,${shot.toString('base64')}"></div>
  </div></body></html>`
}

await main()
