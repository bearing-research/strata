import { onMounted, onBeforeUnmount, ref, watch, type Ref } from 'vue'
import { Compartment, EditorState } from '@codemirror/state'
import {
  EditorView,
  keymap,
  lineNumbers,
  highlightActiveLine,
  highlightActiveLineGutter,
} from '@codemirror/view'
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands'
import { markdown } from '@codemirror/lang-markdown'
import { python } from '@codemirror/lang-python'
import { oneDark } from '@codemirror/theme-one-dark'
import {
  syntaxHighlighting,
  defaultHighlightStyle,
  bracketMatching,
  StreamLanguage,
} from '@codemirror/language'
// Legacy-modes package ships CM5 modes ported to CM6 — no first-party
// ``@codemirror/lang-r`` exists yet, so wrap the legacy R mode in
// ``StreamLanguage.define()`` to get syntax highlighting + indent in
// R cells. Same pattern any CM6 ecosystem doc recommends for R.
import { r as rLegacyMode } from '@codemirror/legacy-modes/mode/r'
import { closeBrackets } from '@codemirror/autocomplete'
import type { CellLanguage } from '../types/notebook'
import { useTheme } from './useTheme'

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
// Dark theme: oneDark (bundled) + a gutter tweak to match the Mocha base.
// Light theme: hand-rolled minimal theme with Catppuccin Latte colors so we
// don't pull in an extra dep. `defaultHighlightStyle` still provides syntax
// colors — both themes share it, and it reads well on either background.

const darkTheme = [
  oneDark,
  EditorView.theme({
    '.cm-gutters': { backgroundColor: '#1e1e2e', border: 'none' },
  }),
]

const lightTheme = EditorView.theme(
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
)

export function useCodemirror(
  container: Ref<HTMLElement | null>,
  opts: {
    initialDoc?: string
    language?: CellLanguage
    onUpdate?: (doc: string) => void
    onRun?: () => void
    onRerun?: () => void
  } = {},
) {
  const view = ref<EditorView | null>(null)
  let suppressNextUpdate = false

  // One compartment per editor instance — the compartment lets us swap the
  // theme with view.dispatch({ effects: themeCompartment.reconfigure(...) })
  // instead of rebuilding the EditorState.
  const themeCompartment = new Compartment()
  const { resolved } = useTheme()

  function themeFor(mode: 'light' | 'dark') {
    return mode === 'light' ? lightTheme : darkTheme
  }

  onMounted(() => {
    if (!container.value) return

    // Per-language CodeMirror extension. ``prompt`` cells render as plain
    // text (no syntax highlighting — the body is a template, not code).
    // ``markdown`` uses the official lang-markdown package; ``r`` wraps
    // the legacy CM5 R mode via ``StreamLanguage`` (no first-party
    // ``@codemirror/lang-r`` exists yet); everything else (``python``,
    // ``sql``, future additions) falls through to Python highlighting,
    // which is the closest visual fit.
    const langExt =
      opts.language === 'markdown'
        ? markdown()
        : opts.language === 'prompt'
          ? []
          : opts.language === 'r'
            ? StreamLanguage.define(rLegacyMode)
            : python()

    const runKeymap = keymap.of([
      {
        key: 'Shift-Enter',
        run: () => {
          opts.onRun?.()
          return true
        },
      },
      {
        key: 'Mod-Shift-Enter',
        run: () => {
          opts.onRerun?.()
          return true
        },
      },
    ])

    const updateListener = EditorView.updateListener.of((update) => {
      if (update.docChanged && !suppressNextUpdate) {
        opts.onUpdate?.(update.state.doc.toString())
      }
    })

    const state = EditorState.create({
      doc: opts.initialDoc ?? '',
      extensions: [
        lineNumbers(),
        highlightActiveLine(),
        highlightActiveLineGutter(),
        history(),
        bracketMatching(),
        closeBrackets(),
        syntaxHighlighting(defaultHighlightStyle),
        langExt,
        themeCompartment.of(themeFor(resolved.value)),
        keymap.of([...defaultKeymap, ...historyKeymap]),
        runKeymap,
        updateListener,
        EditorView.theme({
          '&': { fontSize: '13px' },
          '.cm-content': {
            fontFamily: '"JetBrains Mono", "Fira Code", monospace',
            padding: '8px 0',
          },
          '.cm-scroller': { overflow: 'auto' },
        }),
      ],
    })

    view.value = new EditorView({ state, parent: container.value })
  })

  // Swap theme live when the user flips the toggle — no editor rebuild.
  watch(resolved, (mode) => {
    const v = view.value
    if (!v) return
    v.dispatch({ effects: themeCompartment.reconfigure(themeFor(mode)) })
  })

  onBeforeUnmount(() => {
    view.value?.destroy()
  })

  function setDoc(doc: string) {
    const v = view.value
    if (!v) return
    if (v.state.doc.toString() === doc) return
    suppressNextUpdate = true
    v.dispatch({
      changes: { from: 0, to: v.state.doc.length, insert: doc },
    })
    suppressNextUpdate = false
  }

  return { view, setDoc }
}
