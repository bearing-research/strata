import { Prec, type Extension } from '@codemirror/state'
import { keymap } from '@codemirror/view'
import { defaultKeymap, historyKeymap } from '@codemirror/commands'

export interface RunHandlers {
  onRun?: () => void
  onRerun?: () => void
}

/**
 * The cell editor's key bindings: run and rerun, then CodeMirror's defaults.
 *
 * The run bindings take high precedence because the default keymap binds
 * Enter with a Shift variant (insert a newline), and between two keymaps of
 * the same precedence the earlier one wins. Listed after it, Shift+Enter
 * inserted a newline and never ran the cell.
 */
export function editorKeymaps(handlers: RunHandlers): Extension[] {
  return [
    Prec.high(
      keymap.of([
        {
          key: 'Shift-Enter',
          run: () => {
            handlers.onRun?.()
            return true
          },
        },
        {
          key: 'Mod-Shift-Enter',
          run: () => {
            handlers.onRerun?.()
            return true
          },
        },
      ]),
    ),
    keymap.of([...defaultKeymap, ...historyKeymap]),
  ]
}
