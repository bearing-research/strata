import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import * as lucide from '@lucide/vue'

import { CELL_STATES, cellState, type CellState } from './cellState.ts'

const BLOCKING = [{ severity: 'error' }]

test('each status reads as its state', () => {
  const cases: [Parameters<typeof cellState>[0], CellState][] = [
    [{ status: 'idle' }, 'idle'],
    [{ status: 'queued' }, 'queued'],
    [{ status: 'running' }, 'running'],
    [{ status: 'running', remoteWorkerName: 'gpu' }, 'dispatching'],
    [{ status: 'ready' }, 'ready'],
    [{ status: 'ready', output: { cacheHit: true } }, 'cached'],
    [{ status: 'stale', output: { cacheHit: true } }, 'stale'],
    [{ status: 'error' }, 'error'],
  ]
  for (const [cell, state] of cases) assert.equal(cellState(cell), state)
})

test('an annotation error blocks a cell, and a warning does not', () => {
  for (const status of ['idle', 'ready', 'stale', 'error']) {
    assert.equal(cellState({ status, annotationDiagnostics: BLOCKING }), 'blocked')
  }
  assert.equal(
    cellState({ status: 'ready', annotationDiagnostics: [{ severity: 'warn' }] }),
    'ready',
  )
})

test('a run in flight says so, even with an annotation error since', () => {
  assert.equal(cellState({ status: 'running', annotationDiagnostics: BLOCKING }), 'running')
})

test('DESIGN.md names every state with its token, icon and label', () => {
  const design = readFileSync(new URL('../../DESIGN.md', import.meta.url), 'utf8')
  const style = readFileSync(new URL('../style.css', import.meta.url), 'utf8')
  const rows = design
    .split('\n')
    .filter((line) => line.startsWith('|'))
    .map((line) => line.split('|').map((part) => part.trim()))
  for (const [state, { label, icon }] of Object.entries(CELL_STATES)) {
    const row = rows.find((cells) => cells[1] === state)
    assert.ok(row, `no row for ${state}`)
    assert.ok(row.includes(`\`--state-${state}\``), `${state}: token`)
    const names = Object.entries(lucide).flatMap(([name, value]) => (value === icon ? [name] : []))
    assert.ok(
      names.some((name) => row.some((cell) => cell.startsWith(`\`${name}\``))),
      `${state}: icon`,
    )
    assert.ok(row.includes(label), `${state}: label`)
    assert.ok(style.includes(`--state-${state}: `), `${state}: color token`)
    assert.ok(style.includes(`--state-${state}-tint: `), `${state}: tint token`)
    assert.ok(style.includes(`.state-${state} {`), `${state}: class`)
  }
})
