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
 * Run bindings need high precedence: the default keymap binds Shift+Enter to
 * insert a newline, and at equal precedence the earlier keymap wins.
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

/**
 * Undo and redo as the shortcuts modal names them. CodeMirror's history keymap
 * binds redo to Mod+Shift+Z on a Mac and Mod+Y elsewhere.
 */
export function historyShortcuts(mac: boolean): { undo: string; redo: string } {
  return { undo: 'Mod + Z', redo: mac ? 'Mod + Shift + Z' : 'Mod + Y' }
}
