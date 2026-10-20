import { onMounted, onBeforeUnmount, ref, watch, type Ref } from 'vue'
import { Compartment, EditorState } from '@codemirror/state'
import {
  EditorView,
  lineNumbers,
  highlightActiveLine,
  highlightActiveLineGutter,
} from '@codemirror/view'
import { history } from '@codemirror/commands'
import { python } from '@codemirror/lang-python'
import { bracketMatching, StreamLanguage } from '@codemirror/language'
// No first-party ``@codemirror/lang-r`` exists; wrap the legacy CM5 mode in
// ``StreamLanguage.define()``.
import { r as rLegacyMode } from '@codemirror/legacy-modes/mode/r'
import { closeBrackets } from '@codemirror/autocomplete'
import type { CellLanguage } from '../types/notebook'
import { editorKeymaps } from './editorKeymaps'
import { editorTheme } from './editorThemes'
import { useTheme } from './useTheme'

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

  // Compartments swap the theme or language without rebuilding the EditorState.
  const themeCompartment = new Compartment()
  // lang-markdown is 490 kB of the editor's 563 kB (it carries nested grammars
  // for fenced code), so it loads only when a markdown cell mounts. Until then
  // the cell renders as plain text.
  const langCompartment = new Compartment()
  const { resolved } = useTheme()

  onMounted(() => {
    if (!container.value) return

    // ``prompt`` bodies are templates, so plain text. Anything without its own
    // mode (``sql`` included) gets Python highlighting, the closest fit.
    const langExt =
      opts.language === 'markdown'
        ? []
        : opts.language === 'prompt'
          ? []
          : opts.language === 'r'
            ? StreamLanguage.define(rLegacyMode)
            : python()

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
        langCompartment.of(langExt),
        themeCompartment.of(editorTheme(resolved.value)),
        ...editorKeymaps(opts),
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

    if (opts.language === 'markdown') {
      // The guard covers a cell unmounted before the grammar arrives:
      // dispatching into a destroyed view throws.
      void import('@codemirror/lang-markdown')
        .then(({ markdown }) => {
          const v = view.value
          if (!v) return
          v.dispatch({ effects: langCompartment.reconfigure(markdown()) })
        })
        .catch((error) => {
          // The cell still works; warn so plain-text markdown has an explanation.
          console.warn('Markdown syntax highlighting could not be loaded', error)
        })
    }
  })

  watch(resolved, (mode) => {
    const v = view.value
    if (!v) return
    v.dispatch({ effects: themeCompartment.reconfigure(editorTheme(mode)) })
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
