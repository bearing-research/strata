import assert from 'node:assert/strict'
import test from 'node:test'

import { defaultHighlightStyle, highlightingFor } from '@codemirror/language'
import { EditorState } from '@codemirror/state'
import { oneDarkHighlightStyle } from '@codemirror/theme-one-dark'
import { tags } from '@lezer/highlight'

import { editorTheme } from './editorThemes.ts'

/** The classes every active highlighter gives a keyword in a *mode* editor. */
function keywordClasses(mode: 'light' | 'dark'): string | null {
  const state = EditorState.create({ extensions: editorTheme(mode) })
  return highlightingFor(state, [tags.keyword])
}

test('a dark editor colors keywords with the dark palette alone', () => {
  // The light palette's keyword purple is unreadable on the dark background.
  assert.equal(keywordClasses('dark'), oneDarkHighlightStyle.style([tags.keyword]))
})

test('a light editor colors keywords with the light palette alone', () => {
  assert.equal(keywordClasses('light'), defaultHighlightStyle.style([tags.keyword]))
})
