## Sanitization checks

Markdown cells (and `Markdown(...)` outputs) are user-controlled and
rendered via `v-html`, so the renderer runs every output through
DOMPurify. Each item below should appear as **literal text or a no-op**
,  never executed, never producing a popup, never with the dangerous
attribute reaching the DOM.

### Script injection

Below should render as plain text inside a paragraph, not run:

<script>alert('XSS via raw script tag')</script>

### Inline HTML attempts

The renderer is configured with `html: false`, so any inline HTML
we accidentally let through still gets sanitized. The next line should
remain as a paragraph with the angle-bracket text visible:

<img src="x" onerror="alert('attribute XSS')" />

### `javascript:` URLs

A markdown link with a `javascript:` URL should either render as
inert text or be stripped entirely, never as a clickable executor:

[click here please](javascript:alert('not allowed'))

### `on*` handler attempts inside markdown

Markdown doesn't have a syntax for HTML attributes, but if a future
extension allowed `{onclick=...}` style attribute lists, it must not
plumb through.

### Data URLs

Data URLs in links are sanitized too, DOMPurify drops the href:

[image data url](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)
