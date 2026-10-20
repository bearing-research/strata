import assert from 'node:assert/strict'
import test from 'node:test'

import { historyKeymap, redo, undo } from '@codemirror/commands'
import { EditorState } from '@codemirror/state'
import { keymap, type KeyBinding } from '@codemirror/view'

import { editorKeymaps, historyShortcuts } from './editorKeymaps.ts'

/** The binding CodeMirror consults first for a Shift+Enter keypress. */
function firstShiftEnter(bindings: readonly KeyBinding[]): KeyBinding | undefined {
  return bindings.find((b) => b.key === 'Shift-Enter' || (b.key === 'Enter' && b.shift))
}

test('Shift+Enter runs the cell rather than inserting a newline', () => {
  let ran = 0
  const state = EditorState.create({ extensions: editorKeymaps({ onRun: () => ran++ }) })
  const binding = firstShiftEnter(state.facet(keymap).flat())

  assert.equal(binding?.key, 'Shift-Enter')
  binding?.run?.(undefined as never)
  assert.equal(ran, 1)
})

test('Mod+Shift+Enter reruns the cell', () => {
  let reran = 0
  const state = EditorState.create({ extensions: editorKeymaps({ onRerun: () => reran++ }) })
  const binding = state
    .facet(keymap)
    .flat()
    .find((b) => b.key === 'Mod-Shift-Enter')

  binding?.run?.(undefined as never)
  assert.equal(reran, 1)
})

/** CodeMirror's key names for a command on a platform, as the modal spells them. */
function boundKeys(command: unknown, platform: 'mac' | 'win'): string[] {
  return historyKeymap
    .filter((b) => b.run === command)
    .map((b) => b[platform] ?? b.key)
    .filter((key): key is string => key !== undefined)
    .map((key) =>
      key
        .split('-')
        .map((part) => (part.length === 1 ? part.toUpperCase() : part))
        .join(' + '),
    )
}

test('the shortcuts modal names the undo and redo keys the editor binds', () => {
  for (const [platform, mac] of [
    ['mac', true],
    ['win', false],
  ] as const) {
    const shown = historyShortcuts(mac)
    assert.ok(boundKeys(undo, platform).includes(shown.undo), `${platform} undo: ${shown.undo}`)
    assert.ok(boundKeys(redo, platform).includes(shown.redo), `${platform} redo: ${shown.redo}`)
  }
})
