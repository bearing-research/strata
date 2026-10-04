#!/usr/bin/env node

import fs from 'node:fs/promises'
import process from 'node:process'
import { chromium } from 'playwright'

const DEFAULT_BASE_URL = process.env.STRATA_BENCHMARK_BASE_URL || 'http://127.0.0.1:8765'
const DEFAULT_PARENT_PATH =
  process.env.STRATA_BENCHMARK_PARENT_PATH || '/tmp/strata-benchmark-notebooks'
const DEFAULT_TIMEOUT_MS = 45_000

function printUsage() {
  console.log(`Notebook browser benchmark

Usage:
  npm run benchmark:notebook -- --base-url http://127.0.0.1:8765

Options:
  --base-url URL         App origin to benchmark (default: ${DEFAULT_BASE_URL})
  --parent-path PATH     Parent directory for created notebooks (default: ${DEFAULT_PARENT_PATH})
  --python VERSION       Requested Python version for notebook creation
  --iterations N         Number of create/open iterations to run (default: 1)
  --timeout-ms N         Per-step timeout in milliseconds (default: ${DEFAULT_TIMEOUT_MS})
  --headed               Run Chromium headed instead of headless
  --json PATH            Write full benchmark result JSON to a file
  --help                 Show this message

Examples:
  npm run benchmark:notebook -- --base-url http://127.0.0.1:8765
  npm run benchmark:notebook -- --base-url https://strata.example.com --iterations 3 --json /tmp/notebook-bench.json
`)
}

function parseArgs(argv) {
  const options = {
    baseUrl: DEFAULT_BASE_URL,
    parentPath: DEFAULT_PARENT_PATH,
    pythonVersion: null,
    iterations: 1,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    headless: true,
    jsonPath: null,
  }

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    switch (arg) {
      case '--base-url':
        options.baseUrl = argv[++i]
        break
      case '--parent-path':
        options.parentPath = argv[++i]
        break
      case '--python':
        options.pythonVersion = argv[++i]
        break
      case '--iterations':
        options.iterations = Number(argv[++i] || '1')
        break
      case '--timeout-ms':
        options.timeoutMs = Number(argv[++i] || String(DEFAULT_TIMEOUT_MS))
        break
      case '--headed':
        options.headless = false
        break
      case '--json':
        options.jsonPath = argv[++i]
        break
      case '--help':
      case '-h':
        printUsage()
        process.exit(0)
      default:
        throw new Error(`Unknown argument: ${arg}`)
    }
  }

  if (!Number.isFinite(options.iterations) || options.iterations < 1) {
    throw new Error(`Invalid --iterations value: ${options.iterations}`)
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 1) {
    throw new Error(`Invalid --timeout-ms value: ${options.timeoutMs}`)
  }

  options.baseUrl = options.baseUrl.replace(/\/+$/, '')
  return options
}

function parseServerTiming(header) {
  if (!header) return {}
  const result = {}
  for (const part of header.split(',')) {
    const trimmed = part.trim()
    if (!trimmed) continue
    const [namePart, ...params] = trimmed.split(';').map((value) => value.trim())
    if (!namePart) continue
    const entry = {}
    for (const param of params) {
      const [key, value] = param.split('=')
      if (!key || value == null) continue
      if (key === 'dur') {
        const numeric = Number(value)
        entry.durationMs = Number.isFinite(numeric) ? numeric : value
      } else if (key === 'desc') {
        entry.description = value.replace(/^"|"$/g, '')
      } else {
        entry[key] = value.replace(/^"|"$/g, '')
      }
    }
    result[namePart] = entry
  }
  return result
}

async function collectMeasures(page) {
  return page.evaluate(() => {
    const prefix = 'strata:notebook:'
    const entries = performance
      .getEntriesByType('measure')
      .filter((entry) => entry.name.endsWith('_ms'))
      .map((entry) => ({
        name: entry.name.startsWith(prefix) ? entry.name.slice(prefix.length) : entry.name,
        duration: Number(entry.duration.toFixed(2)),
      }))

    const latestByName = {}
    for (const entry of entries) {
      latestByName[entry.name] = entry.duration
    }
    return latestByName
  })
}

async function waitForOptionalMeasure(page, measureName, timeoutMs) {
  try {
    await page.waitForFunction(
      (name) => {
        const entryName = `strata:notebook:${name}`
        return performance.getEntriesByName(entryName, 'measure').length > 0
      },
      measureName,
      { timeout: timeoutMs },
    )
  } catch {
    // The benchmark should still succeed even if the optional measure is not
    // present before timeout.
  }
}

async function waitForNotebookReady(page, timeoutMs) {
  await page.waitForURL(/#\/notebook\//, { timeout: timeoutMs })
  await page.getByTestId('notebook-page').waitFor({ state: 'visible', timeout: timeoutMs })
  await page.getByTestId('notebook-cells-panel').waitFor({ state: 'visible', timeout: timeoutMs })
  await page.getByTestId('notebook-add-cell').waitFor({ state: 'visible', timeout: timeoutMs })
}

async function goHome(page, baseUrl, timeoutMs) {
  await page.goto(`${baseUrl}/#/`, { waitUntil: 'domcontentloaded', timeout: timeoutMs })
  await page.getByTestId('home-page').waitFor({ state: 'visible', timeout: timeoutMs })
}

async function runCreateFlow(page, options, notebookName) {
  await goHome(page, options.baseUrl, options.timeoutMs)
  await page.getByTestId('action-new-notebook').click()
  await page
    .getByTestId('new-notebook-form')
    .waitFor({ state: 'visible', timeout: options.timeoutMs })
  await page.getByTestId('new-notebook-name').fill(notebookName)
  await page.getByTestId('new-notebook-parent-path').fill(options.parentPath)

  if (options.pythonVersion) {
    const pythonSelect = page.getByTestId('new-notebook-python-version')
    const isDisabled = await pythonSelect.isDisabled()
    if (!isDisabled) {
      await pythonSelect.selectOption(options.pythonVersion)
    }
  }

  const startedAt = Date.now()
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' && response.url().endsWith('/v1/notebooks/create'),
    { timeout: options.timeoutMs },
  )
  await page.getByTestId('create-notebook-submit').click()
  const response = await responsePromise
  await waitForNotebookReady(page, options.timeoutMs)
  await waitForOptionalMeasure(page, 'store_session_ws_connect_ms', 2_000)
  const finishedAt = Date.now()

  const body = await response.json().catch(() => ({}))
  return {
    status: response.status(),
    notebookPath: typeof body.path === 'string' ? body.path : null,
    sessionId: typeof body.session_id === 'string' ? body.session_id : null,
    wallClockMs: finishedAt - startedAt,
    measures: await collectMeasures(page),
    serverTiming: parseServerTiming(response.headers()['server-timing']),
    finalUrl: page.url(),
  }
}

async function runOpenFlow(page, options, notebookPath) {
  await goHome(page, options.baseUrl, options.timeoutMs)
  await page.getByTestId('action-open-notebook').click()
  await page
    .getByTestId('open-notebook-form')
    .waitFor({ state: 'visible', timeout: options.timeoutMs })
  // The Open Existing UI is now a discovered-notebook list keyed by
  // absolute path (see HomePage.vue, loadDiscoveredNotebooks). Wait
  // for the list to populate then click the entry whose data-
  // notebook-path matches the target.
  const targetItem = page.locator(
    `[data-testid="open-notebook-item"][data-notebook-path="${notebookPath}"]`,
  )
  await targetItem.waitFor({ state: 'visible', timeout: options.timeoutMs })

  const startedAt = Date.now()
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' && response.url().endsWith('/v1/notebooks/open'),
    { timeout: options.timeoutMs },
  )
  await targetItem.click()
  const response = await responsePromise
  await waitForNotebookReady(page, options.timeoutMs)
  await waitForOptionalMeasure(page, 'store_session_ws_connect_ms', 2_000)
  const finishedAt = Date.now()

  const body = await response.json().catch(() => ({}))
  return {
    status: response.status(),
    notebookPath: typeof body.path === 'string' ? body.path : notebookPath,
    sessionId: typeof body.session_id === 'string' ? body.session_id : null,
    wallClockMs: finishedAt - startedAt,
    measures: await collectMeasures(page),
    serverTiming: parseServerTiming(response.headers()['server-timing']),
    finalUrl: page.url(),
  }
}

function average(values) {
  if (values.length === 0) return null
  const total = values.reduce((sum, value) => sum + value, 0)
  return Number((total / values.length).toFixed(2))
}

function summarizeResults(results) {
  const createFlows = results.map((result) => result.create)
  const openFlows = results.map((result) => result.open)
  const averageMeasure = (flows, measureName) =>
    average(
      flows.map((flow) => flow.measures[measureName]).filter((value) => typeof value === 'number'),
    )

  return {
    iterations: results.length,
    create: {
      wallClockMs: average(createFlows.map((flow) => flow.wallClockMs)),
      createRequestMs: averageMeasure(createFlows, 'create_request_ms'),
      createRouteMs: averageMeasure(createFlows, 'create_route_ms'),
      createTotalMs: averageMeasure(createFlows, 'create_total_ms'),
      notebookMountToReadyMs: averageMeasure(createFlows, 'notebook_mount_to_ready_ms'),
      connectTotalMs: averageMeasure(createFlows, 'connect_total_ms'),
      storeSessionWsConnectMs: averageMeasure(createFlows, 'store_session_ws_connect_ms'),
    },
    open: {
      wallClockMs: average(openFlows.map((flow) => flow.wallClockMs)),
      openRequestMs: averageMeasure(openFlows, 'open_request_ms'),
      openRouteMs: averageMeasure(openFlows, 'open_route_ms'),
      openTotalMs: averageMeasure(openFlows, 'open_total_ms'),
      notebookMountToReadyMs: averageMeasure(openFlows, 'notebook_mount_to_ready_ms'),
      connectTotalMs: averageMeasure(openFlows, 'connect_total_ms'),
      storeSessionWsConnectMs: averageMeasure(openFlows, 'store_session_ws_connect_ms'),
    },
  }
}

function printFlowResult(label, flow) {
  console.log(`${label}:`)
  console.log(`  status: ${flow.status}`)
  console.log(`  notebook_path: ${flow.notebookPath}`)
  console.log(`  session_id: ${flow.sessionId}`)
  console.log(`  wall_clock_ms: ${flow.wallClockMs}`)

  const measureKeys = Object.keys(flow.measures).sort()
  if (measureKeys.length) {
    console.log('  measures:')
    for (const key of measureKeys) {
      console.log(`    ${key}: ${flow.measures[key]}`)
    }
  }

  const timingKeys = Object.keys(flow.serverTiming)
  if (timingKeys.length) {
    console.log('  server_timing:')
    for (const key of timingKeys) {
      const value = flow.serverTiming[key]
      const duration = value.durationMs ?? 'n/a'
      console.log(`    ${key}: ${duration}ms`)
    }
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  const browser = await chromium.launch({ headless: options.headless })
  const results = []

  try {
    for (let index = 0; index < options.iterations; index += 1) {
      const runId = `${Date.now()}-${index + 1}`
      const notebookName = `benchmark-${runId}`
      console.log(`\nIteration ${index + 1}/${options.iterations}`)

      const createPage = await browser.newPage()
      const create = await runCreateFlow(createPage, options, notebookName)
      await createPage.close()

      if (!create.notebookPath) {
        throw new Error('Create flow did not return a notebook path')
      }

      const openPage = await browser.newPage()
      const open = await runOpenFlow(openPage, options, create.notebookPath)
      await openPage.close()

      const iterationResult = { iteration: index + 1, create, open }
      results.push(iterationResult)

      printFlowResult('Create', create)
      printFlowResult('Open', open)
    }
  } finally {
    await browser.close()
  }

  const summary = summarizeResults(results)
  console.log('\nSummary:')
  console.log(JSON.stringify(summary, null, 2))

  if (options.jsonPath) {
    await fs.writeFile(
      options.jsonPath,
      JSON.stringify(
        {
          baseUrl: options.baseUrl,
          parentPath: options.parentPath,
          pythonVersion: options.pythonVersion,
          summary,
          results,
        },
        null,
        2,
      ),
      'utf8',
    )
    console.log(`\nWrote benchmark JSON to ${options.jsonPath}`)
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error)
  if (message.includes("Executable doesn't exist")) {
    console.error(
      'Playwright Chromium is not installed. Run `npx playwright install chromium` in frontend/.',
    )
  }
  console.error(message)
  process.exit(1)
})
