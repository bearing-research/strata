import assert from 'node:assert/strict'
import test from 'node:test'

import {
  appendConsole,
  applyConsoleChunk,
  type CellConsole,
  replaceConsole,
  startConsoleRun,
  startsRun,
} from './consoleChunk.ts'

test("a run's first chunk replaces the last run's console", () => {
  assert.equal(applyConsoleChunk('epoch 9 of the last run\n', 'epoch 1\n', 0), 'epoch 1\n')
})

test('later chunks append', () => {
  assert.equal(applyConsoleChunk('epoch 1\n', 'epoch 2\n', 1), 'epoch 1\nepoch 2\n')
})

test('console sent when the cell finishes appends', () => {
  assert.equal(applyConsoleChunk('epoch 1\n', 'done\n', null), 'epoch 1\ndone\n')
  assert.equal(applyConsoleChunk(undefined, 'done\n', undefined), 'done\n')
})

// Frames as the store's handlers apply them; a local run sends `running`, one
// `chunk_seq: null` frame per stream that printed, then the result.
type Frame = [string, Record<string, unknown>]

function apply(cell: CellConsole, frames: Frame[]): void {
  for (const [type, payload] of frames) {
    if (type === 'cell_status' && startsRun(payload)) startConsoleRun(cell)
    else if (type === 'cell_console') appendConsole(cell, payload)
    else if (type === 'cell_output' || type === 'cell_error') replaceConsole(cell, payload)
  }
}

function localRun(result: 'cell_output' | 'cell_error', stdout: string, stderr = ''): Frame[] {
  const frames: Frame[] = [['cell_status', { cell_id: 'a', status: 'running' }]]
  if (stdout) frames.push(['cell_console', { stream: 'stdout', text: stdout, chunk_seq: null }])
  if (stderr) frames.push(['cell_console', { stream: 'stderr', text: stderr, chunk_seq: null }])
  frames.push([result, { cell_id: 'a', stdout, stderr }])
  return frames
}

test("a failing local cell run three times shows only the last run's console", () => {
  const cell: CellConsole = {}
  for (let i = 0; i < 3; i++) apply(cell, localRun('cell_error', 'world\n', 'Traceback\n'))
  assert.equal(cell.consoleStdout, 'world\n')
  assert.equal(cell.consoleStderr, 'Traceback\n')
})

test("a failure that never reaches the harness drops the last run's console", () => {
  const cell: CellConsole = {}
  apply(cell, localRun('cell_output', 'hello\n'))
  apply(cell, [
    ['cell_status', { cell_id: 'a', status: 'running' }],
    ['cell_error', { cell_id: 'a', error: 'worker unreachable' }],
  ])
  assert.equal(cell.consoleStdout, '')
})

test('the result replaces the console streamed during the run', () => {
  const cell: CellConsole = {}
  apply(cell, [
    ['cell_status', { cell_id: 'a', status: 'running' }],
    ['cell_console', { stream: 'stdout', text: 'epoch 1\n', chunk_seq: 0 }],
    ['cell_output', { cell_id: 'a', stdout: 'epoch 1\nepoch 2\n', stderr: '' }],
  ])
  assert.equal(cell.consoleStdout, 'epoch 1\nepoch 2\n')
})

test("a remote cell's phase frames keep the console streamed so far", () => {
  const cell: CellConsole = {}
  apply(cell, [
    ['cell_status', { cell_id: 'a', status: 'running', remote_worker: 'pool' }],
    ['cell_status', { cell_id: 'a', status: 'running', remote_build_state: 'starting' }],
    ['cell_console', { stream: 'stdout', text: 'epoch 1\n', chunk_seq: 0 }],
    ['cell_status', { cell_id: 'a', status: 'running', remote_build_state: 'running' }],
    ['cell_console', { stream: 'stdout', text: 'epoch 2\n', chunk_seq: 1 }],
  ])
  assert.equal(cell.consoleStdout, 'epoch 1\nepoch 2\n')
})
