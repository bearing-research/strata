/**
 * Markdown to sanitized HTML for markdown cells and ``Markdown(...)`` outputs.
 *
 * Both are user-controlled and rendered via ``v-html``, so DOMPurify is
 * mandatory: without it a malicious notebook could inject `<script>`.
 */

import DOMPurifyFactory, { type Config as DOMPurifyConfig } from 'dompurify'
import MarkdownIt from 'markdown-it'

// In a browser the default export is bound to ``window``. In Node (unit tests)
// it is a factory with no DOM to bind, so use a no-op shim; the HTML never
// reaches a DOM there, and the browser still gets the real DOMPurify.
type DOMPurifyLike = { sanitize: (html: string, cfg?: DOMPurifyConfig) => string }

const purify: DOMPurifyLike =
  typeof (DOMPurifyFactory as unknown as DOMPurifyLike).sanitize === 'function'
    ? (DOMPurifyFactory as unknown as DOMPurifyLike)
    : { sanitize: (html: string) => html }

const md = new MarkdownIt({
  // Sanitization would strip most inline HTML anyway; off is predictable.
  html: false,
  linkify: true,
  // Smart quotes break copy-paste of code identifiers in prose.
  typographer: false,
  // Single newlines become <br> so existing outputs don't reflow.
  breaks: true,
})

// Links open in a new tab with noopener/noreferrer.
const defaultLinkOpen =
  md.renderer.rules.link_open ||
  function (tokens, idx, options, _env, self) {
    return self.renderToken(tokens, idx, options)
  }

md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  const token = tokens[idx]
  const targetIdx = token.attrIndex('target')
  if (targetIdx < 0) {
    token.attrPush(['target', '_blank'])
  } else {
    token.attrs![targetIdx][1] = '_blank'
  }
  const relIdx = token.attrIndex('rel')
  if (relIdx < 0) {
    token.attrPush(['rel', 'noreferrer noopener'])
  } else {
    token.attrs![relIdx][1] = 'noreferrer noopener'
  }
  return defaultLinkOpen(tokens, idx, options, env, self)
}

// Allow the target/rel set above; the defaults strip scripts and handlers.
const PURIFY_CONFIG: DOMPurifyConfig = {
  ADD_ATTR: ['target', 'rel'],
}

export function renderMarkdownToHtml(markdown: string): string {
  if (!markdown) return ''
  const rawHtml = md.render(markdown)
  return purify.sanitize(rawHtml, PURIFY_CONFIG)
}
