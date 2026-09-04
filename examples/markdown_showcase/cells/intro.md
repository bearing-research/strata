# Markdown Showcase

This notebook exercises every markdown rendering path in Strata. It's
deliberately content-only, there's no real "computation" to run. Use it
to verify that the renderer handles the cases listed below, and as a
reference when you want to write a documentation cell.

**Two surfaces are tested:**

1. **Markdown cells** (this cell, and the next several). Source is raw
   markdown; preview renders by default; click anywhere on the preview
   to swap in the editor; blur to render again.
2. **Dynamic markdown output** from Python cells via `Markdown(...)`
   (the last two cells). Same rendering pipeline, different entry
   point, useful when you want a code cell to emit a formatted report.

If anything in the next few cells doesn't render the way you'd expect,
that's a renderer bug worth filing.
