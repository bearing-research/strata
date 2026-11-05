# Widget and app view annotations

## `@live`

On a [widget cell](../../guides/cells/widget.md#widget-cells), `# @live` makes a control change
auto-run the cheap downstream cells instead of only marking them stale.
`# @live off` turns it back off. See [Live mode](../../guides/cells/widget.md#live-mode).

## `@app hide`

`# @app hide` keeps a cell out of the read-only [app view](../../guides/share/app-view.md#app-view),
for a setup cell whose output is noise. It applies to any cell.
In a markdown cell it is the only annotation read: a heading such as
`# @after` there is prose.
