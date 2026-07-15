# Keyboard Shortcuts

Press **?** anywhere in the notebook (outside a code editor) to show the shortcuts modal.

## Editor Shortcuts

| Shortcut | Action |
|----------|--------|
| ++shift+enter++ | Run the current cell |
| ++ctrl+z++ | Undo (in editor) |
| ++ctrl+shift+z++ | Redo (in editor) |
| ++ctrl+a++ | Select all (in editor) |
| ++ctrl+d++ | Select next occurrence |

## Notebook Shortcuts

| Shortcut | Action |
|----------|--------|
| ++question++ | Show keyboard shortcuts help |
| ++escape++ | Close modal / dialog |

## Cell Actions (Buttons)

These are available in the cell gutter (visible on hover):

| Button | Action |
|--------|--------|
| ▶ | Run cell |
| ▲ / ▼ | Move cell up / down |
| + | Add cell below |
| ⎘ | Duplicate cell |
| × | Delete cell |
| 🔍 | Inspect cell inputs (REPL) |

!!! tip
    The cell gutter buttons appear when you hover over a cell. The status dot on the left shows the cell's current state (idle, running, ready, error, stale).

## Not yet bound to keyboard

These operations exist in the UI / WebSocket protocol but don't have
dedicated keyboard shortcuts yet - use the buttons or menu:

- **Run all cells** (WebSocket `notebook_run_all`)
- **Add cell below / above** (use `+` in the cell gutter)
- **Delete cell** (use `×` in the cell gutter, confirms via modal)
- **Reorder cells** (use `▲` / `▼` in the gutter, or drag the cell handle)
- **Navigate cells** (no arrow-key navigation between cells; click to
  focus, or use the editor `↑` / `↓` to move the cursor within a cell)
- **Cancel running cell** (Stop button on running cells)

This list is intentionally explicit so you know what's wired up
versus what's still UI-only. Missing bindings are usually a one-line
addition in `frontend/src/components/KeyboardShortcutsModal.vue`
plus the matching handler - open an issue if a specific shortcut is
high-friction for you.
