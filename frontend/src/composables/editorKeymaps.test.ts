import assert from 'node:assert/strict'
import test from 'node:test'

import { EditorState } from '@codemirror/state'
import { keymap, type KeyBinding } from '@codemirror/view'

import { editorKeymaps } from './editorKeymaps.ts'

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
