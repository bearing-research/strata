import assert from 'node:assert/strict'
import test from 'node:test'

import { injectedBasePath, strataHttpBase, strataWsBase } from './strataBase.ts'

function page(content: string | null): Pick<Document, 'querySelector'> {
  return {
    querySelector: (selector: string) =>
      selector === 'meta[name="strata-base-path"]' && content !== null
        ? ({ getAttribute: (name: string) => (name === 'content' ? content : null) } as Element)
        : null,
  } as Pick<Document, 'querySelector'>
}

test('the base path is read from the meta tag the server injects', () => {
  assert.equal(injectedBasePath(page('/o/acme/lab')), '/o/acme/lab')
  assert.equal(injectedBasePath(page('/o/acme/lab/')), '/o/acme/lab')
})

test('no meta tag, or no document, means the root', () => {
  assert.equal(injectedBasePath(page(null)), '')
  assert.equal(injectedBasePath(undefined), '')
})

test('API and WebSocket URLs carry the base path', () => {
  const http = strataHttpBase(undefined, 'https://app.example.com', '/o/acme/lab')
  assert.equal(http, 'https://app.example.com/o/acme/lab')
  assert.equal(strataWsBase(http), 'wss://app.example.com/o/acme/lab')
  assert.equal(
    strataWsBase(strataHttpBase(undefined, 'http://localhost:8765', '')),
    'ws://localhost:8765',
  )
})

test('VITE_STRATA_URL in development wins over the page', () => {
  assert.equal(
    strataHttpBase('http://localhost:8765', 'http://localhost:5173', '/o/acme/lab'),
    'http://localhost:8765',
  )
})
