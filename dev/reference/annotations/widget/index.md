# Widget and App View Annotations

## `@live`

On a [widget cell](https://bearingresearch.com/strata/dev/guides/cells/widget/#widget-cells), `# @live` makes a control change
auto-run the cheap downstream cells instead of only marking them stale.
`# @live off` turns it back off. See [Live mode](https://bearingresearch.com/strata/dev/guides/cells/widget/#live-mode).

## `@app hide`

`# @app hide` keeps a cell out of the read-only [app view](https://bearingresearch.com/strata/dev/guides/share/app-view/#app-view),
for a setup cell whose output is noise. It applies to any cell.
In a markdown cell it is the only annotation read: a heading such as
`# @after` there is prose.
