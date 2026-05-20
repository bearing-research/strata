import assert from 'node:assert/strict'
import test from 'node:test'

import { renderMarkdownToHtml } from './markdown.ts'

test('renderMarkdownToHtml renders basic markdown blocks', () => {
  const html = renderMarkdownToHtml(`# Title

- one
- **two**

> quote`)

  assert.match(html, /<h1>Title<\/h1>/)
  // markdown-it (CommonMark) emits whitespace between block elements;
  // assert each element appears in order rather than pinning literal HTML.
  assert.match(html, /<ul>[\s\S]*<li>one<\/li>[\s\S]*<li><strong>two<\/strong><\/li>[\s\S]*<\/ul>/)
  assert.match(html, /<blockquote>[\s\S]*<p>quote<\/p>[\s\S]*<\/blockquote>/)
})

test('renderMarkdownToHtml escapes raw html and blocks javascript links', () => {
  const html = renderMarkdownToHtml(
    `<script>alert(1)</script>

[safe](https://example.com)
[bad](javascript:alert(1))`,
  )

  // Inline HTML is escaped (markdown-it ``html: false``) — ``<script>``
  // never reaches the DOM.
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/)
  assert.doesNotMatch(html, /<script>/i)
  // Safe links render as anchors with ``target="_blank" rel="noreferrer noopener"``.
  assert.match(html, /href="https:\/\/example\.com"/)
  // ``javascript:`` URLs are rejected by markdown-it's link validator
  // and the source is left as inert literal text — there should be no
  // anchor tag carrying a ``javascript:`` href.
  assert.doesNotMatch(html, /href="javascript:/)
  assert.doesNotMatch(html, /<a [^>]*javascript:/)
})

test('renderMarkdownToHtml renders fenced code blocks and tables', () => {
  const html = renderMarkdownToHtml(
    [
      '```python',
      'print("hi")',
      '```',
      '',
      '| name | score |',
      '| ---- | ----- |',
      '| alice | 10 |',
    ].join('\n'),
  )

  // Fenced code: ``<code class="language-python">`` with HTML-escaped body.
  // markdown-it appends a trailing newline before ``</code>`` per CommonMark.
  assert.match(html, /<pre><code class="language-python">print\(&quot;hi&quot;\)\n?<\/code><\/pre>/)
  assert.match(html, /<table>/)
  assert.match(html, /<th>name<\/th>/)
  assert.match(html, /<td>alice<\/td>/)
})
