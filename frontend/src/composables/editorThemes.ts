import type { Extension } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { defaultHighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { oneDark } from '@codemirror/theme-one-dark'

// Dark: oneDark, which brings its own syntax colors, plus a gutter tweak for
// the Mocha base. Light: a minimal hand-rolled Catppuccin Latte theme, to avoid
// another dependency, with `defaultHighlightStyle` for syntax colors. Each
// theme owns its highlighter: a light palette left active in dark mode outranks
// oneDark's and paints keywords dark purple on a dark background.

const darkTheme = [
  oneDark,
  EditorView.theme({
    '.cm-gutters': { backgroundColor: '#1e1e2e', border: 'none' },
  }),
]

const lightTheme = [
  EditorView.theme(
    {
      '&': { color: '#4c4f69', backgroundColor: '#ffffff' },
      '.cm-content': { caretColor: '#1e66f5' },
      '.cm-cursor, .cm-dropCursor': { borderLeftColor: '#1e66f5' },
      '&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, ::selection': {
        backgroundColor: '#bcc0cc',
      },
      '.cm-gutters': { backgroundColor: '#e6e9ef', color: '#6c6f85', border: 'none' },
      '.cm-activeLineGutter': { backgroundColor: '#dce0e8' },
      '.cm-activeLine': { backgroundColor: '#e6e9ef' },
    },
    { dark: false },
  ),
  syntaxHighlighting(defaultHighlightStyle),
]

/** The cell editor's theme and syntax colors for *mode*. */
export function editorTheme(mode: 'light' | 'dark'): Extension {
  return mode === 'light' ? lightTheme : darkTheme
}
