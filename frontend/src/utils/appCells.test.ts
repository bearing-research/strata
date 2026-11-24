import assert from 'node:assert/strict'
import test from 'node:test'

import type { Cell } from '../types/notebook.ts'
import { appCells } from './appCells.ts'

function cell(id: string, overrides: Partial<Cell>): Cell {
  return { id, source: '', language: 'python', ...overrides } as Cell
}

const annotations = { env: {}, mounts: [] }

test('a cell marked app-hidden is left out, whatever its source says', () => {
  const shown = cell('a', { language: 'markdown', annotations })
  const hidden = cell('b', {
    language: 'markdown',
    source: 'Notes\n',
    annotations: { ...annotations, appHidden: true },
  })
  // The backend did not read this line, which follows the prose.
  const stray = cell('c', { language: 'markdown', source: 'Notes\n# @app hide\n', annotations })

  assert.deepEqual(
    appCells([shown, hidden, stray]).map((c) => c.id),
    ['a', 'c'],
  )
})

test('a code cell is shown only with a display output', () => {
  const quiet = cell('a', { annotations })
  const drawn = cell('b', {
    annotations,
    displayOutputs: [{ contentType: 'image/png' }] as Cell['displayOutputs'],
  })

  assert.deepEqual(
    appCells([quiet, drawn]).map((c) => c.id),
    ['b'],
  )
})
