import assert from 'node:assert/strict'
import test from 'node:test'

import config from '../../vite.config.ts'

test('the bundle is built with relative asset URLs, so it loads under a base path', () => {
  // Without this the index references /assets/..., which a server at /o/acme/lab cannot serve.
  assert.equal(config.base, './')
})
