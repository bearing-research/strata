import assert from 'node:assert/strict'
import test from 'node:test'

import { embedSnippet } from './embedSnippet.ts'

// Runs the snippet's listener against a stub page and returns the iframe's height
// after one message event.
function heightAfter(snippet: string, event: { origin: string; data: unknown }): string {
  const script = snippet.slice(snippet.indexOf('<script>') + 8, snippet.indexOf('</script>'))
  const frame = { style: { height: '' } }
  let listener: ((e: unknown) => void) | undefined
  new Function('addEventListener', 'document', script)(
    (_type: string, fn: (e: unknown) => void) => (listener = fn),
    { querySelector: () => frame },
  )
  assert.ok(listener, 'the snippet registers a message listener')
  listener(event)
  return frame.style.height
}

test('the snippet frames the app view of the session', () => {
  const snippet = embedSnippet('https://strata.example.com', '', 'abc123')
  assert.match(snippet, /src="https:\/\/strata\.example\.com\/#\/app\/abc123\?embed=1"/)
})

test('under a base path the snippet frames the app view at that path', () => {
  const snippet = embedSnippet('https://app.example.com', '/o/acme/lab', 'abc123')
  assert.match(snippet, /src="https:\/\/app\.example\.com\/o\/acme\/lab\/#\/app\/abc123\?embed=1"/)
  assert.match(snippet, /e\.origin==="https:\/\/app\.example\.com"/)
})

test('the resize listener takes heights only from the Strata origin', () => {
  const snippet = embedSnippet('https://strata.example.com', '', 'abc123')
  const resize = { type: 'strata:embed:resize', height: 640 }

  assert.equal(
    heightAfter(snippet, { origin: 'https://strata.example.com', data: resize }),
    '640px',
  )
  assert.equal(heightAfter(snippet, { origin: 'https://evil.example', data: resize }), '')
})
