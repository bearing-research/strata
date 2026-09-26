# Keyboard Shortcuts

Press **?** anywhere in the notebook (outside a code editor or text field) to
show the shortcuts modal.

On macOS, read ++ctrl++ as ++cmd++ in the shortcuts below.

## Editor Shortcuts

| Shortcut | Action |
|----------|--------|
| ++shift+enter++ | Run the current cell |
| ++ctrl+shift+enter++ | Rerun the current cell, bypassing the cache |
| ++ctrl+z++ | Undo |
| ++ctrl+y++ (++cmd+shift+z++ on macOS) | Redo |
| ++ctrl+a++ | Select all |

The editor also carries CodeMirror's default keymap: ++ctrl+slash++ toggles a
comment, ++ctrl+bracket-left++ / ++ctrl+bracket-right++ change indentation,
and ++alt+arrow-up++ / ++alt+arrow-down++ move the current line.

## Notebook Shortcuts

| Shortcut | Action |
|----------|--------|
| ++question++ | Show or hide the keyboard shortcuts modal |
| ++escape++ | Close the shortcuts modal, or an open menu |

## Panel Shortcuts

| Shortcut | Where | Action |
|----------|-------|--------|
| ++enter++ | Inspect panel input | Evaluate the expression |
| ++ctrl+enter++ | Tests panel | Run the cell's tests |
| ++enter++ | AI assistant input | Send as chat |
| ++shift+enter++ | AI assistant input | Send in agent mode |

## Cell Actions (Buttons)

These are in the cell gutter, visible on hover. The glyphs are the buttons'
actual labels:

| Button | Action |
|--------|--------|
| ▶ | Run cell |
| ↻ | Rerun cell, bypassing the cache |
| ■ | Cancel the running cell (replaces ▶ and ↻ while it runs) |
| ▲ / ▼ | Move cell up / down |
| + | Add cell below |
| ⎘ | Duplicate cell |
| × | Delete cell |
| ▽ / ▷ | Collapse / expand cell |
| 🔍 | Inspect cell inputs (REPL) |
| 🧪 | Unit tests (Python cells) |

!!! tip
    The status glyph at the top of the gutter shows the cell's current state
    (idle, queued, running, ready, stale, error); hover it for details.

## Not yet bound to keyboard

These operations exist in the UI but don't have dedicated keyboard shortcuts
yet. Use the buttons:

- **Run all / rerun all cells** (**▶ Run All** and **↻ Rerun All** in the header)
- **Add cell** (`+` in the cell gutter adds below; **Add cell** in the header
  appends a cell of the kind you pick). There is no "add above".
- **Delete cell** (use `×` in the cell gutter; it deletes immediately, with no
  confirmation, and does nothing when only one cell is left)
- **Reorder cells** (use `▲` / `▼` in the gutter; there is no drag to reorder)
- **Navigate cells** (no arrow-key navigation between cells; click to
  focus, or use the editor `↑` / `↓` to move the cursor within a cell)

This list is intentionally explicit so you know what's wired up
versus what's still UI-only. Missing bindings are usually a one-line
addition to the keymap in `frontend/src/composables/useCodemirror.ts` (or the
global handler in `frontend/src/views/NotebookPage.vue`), plus a row in
`frontend/src/components/KeyboardShortcutsModal.vue`. Open an issue if a
specific shortcut is high-friction for you.
