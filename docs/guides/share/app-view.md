# App view

Click **App** in the notebook header (or visit `/#/app/<sessionId>`) to open the
notebook as a **read-only interactive app**: only widget control panels,
markdown, and display outputs render - no editor, DAG, or toolbars. The
connection is read-only (edits and arbitrary cell runs are rejected server-side),
but viewers can still drive widgets - combined with **⚡ Live** that's the
"tweak a parameter, see the result" dashboard you hand to a stakeholder.

Add `# @app hide` to a cell to keep it out of the app view (e.g. a setup cell
whose output is noise). Run the notebook's cells once in the editor so the app
has outputs to show, then share the app link.

## Embedding the app view in another site

The app view can be dropped into another page - a dashboard, wiki, or internal
portal - as an `<iframe>`. Pick **Embed** from the notebook's **Export** menu to
copy a ready-to-paste snippet, or build it yourself:

```html
<iframe src="http://localhost:8765/#/app/<sessionId>?embed=1"
        title="Strata notebook" style="width:100%;border:0"></iframe>
<script>
  addEventListener('message', (e) => {
    if (e.origin === 'http://localhost:8765' &&
        e.data && e.data.type === 'strata:embed:resize')
      for (const f of document.querySelectorAll('iframe[title="Strata notebook"]'))
        if (f.contentWindow === e.source) f.style.height = e.data.height + 'px'
  })
</script>
```

`?embed=1` drops the standalone chrome (the title bar and **← Edit** link) so the
view blends into the host page, and the embedded app posts its content height to
the parent frame (`{ type: 'strata:embed:resize', height }`) so the optional
listener above sizes the iframe with no inner scrollbar. The listener accepts
heights only from the Strata server's origin, so another frame on the page
cannot resize it, and sizes only the iframe that sent the height, so several
embeds on one page each get their own. Widgets stay live inside
the frame, so with **⚡ Live** on it's a fully interactive embedded panel.

**Cross-origin embedding is opt-in.** By default a notebook is framable only from
its own origin (`Content-Security-Policy: frame-ancestors 'self'`). To embed on
another host, list its origin in `embed_frame_ancestors` (env
`STRATA_EMBED_FRAME_ANCESTORS`), e.g. `https://analytics.example.com`, or `*` to
allow any host. Accepts a comma-separated list or a JSON array.

Every page of the web UI, the editor and the app view alike, also sends
`script-src 'self'; object-src 'none'; base-uri 'self'`, so only Strata's own
bundled scripts run in them: markup that reaches the page
through a cell output or a markdown cell cannot execute script. A proxy that
injects its own inline script into the page needs to replace that policy.

## Framing the editor

A platform that frames the full editor, not only the app view, adds `?framed=1`
(`/#/notebook/<sessionId>?framed=1`). The editor then leaves out what the
parent page already has: Strata's brand and the notebook's name, the **App**,
**Artifacts** and **Logs** links (which would navigate the frame away), the
deployment-mode badge, **Delete Notebook**, and **Back Home** on the reconnect
screen. Reopening the notebook after a lost session keeps `?framed=1`. Running,
editing and export are unchanged. Hiding **Delete Notebook** is not access control: a platform that
wants deletes to go through its own records refuses `DELETE
/v1/notebooks/{session_id}` and `POST /v1/notebooks/delete-by-path` itself. The
same `embed_frame_ancestors` rule decides which hosts may frame it.

A framed editor posts the cell the member is on to the parent page, each time
it changes, so a panel beside the frame can follow it:

```js
window.addEventListener('message', (e) => {
  if (e.source !== frame.contentWindow || e.origin !== strataOrigin) return
  if (e.data && e.data.type === 'strata:frame:focus') {
    // e.data.sessionId, and e.data.cellId, which is null when no cell has focus
  }
})
```
