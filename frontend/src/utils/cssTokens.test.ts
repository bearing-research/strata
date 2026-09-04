import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

import { collectTokens, undefinedTokens } from './cssTokens.ts'

const SRC = join(fileURLToPath(new URL('.', import.meta.url)), '..')
// Styles live in .vue and .css; a .ts file only ever *defines* a variable, via
// an inline style binding. Tests are skipped so their fixtures — deliberately
// undefined names — are not mistaken for the real thing.
const REFERENCE_FILES = ['.vue', '.css']
const DEFINITION_FILES = ['.vue', '.css', '.ts']

function* sources(dir: string, extensions: string[]): Generator<{ path: string; text: string }> {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) {
      yield* sources(path, extensions)
    } else if (!entry.endsWith('.test.ts') && extensions.some((ext) => entry.endsWith(ext))) {
      yield { path: path.slice(SRC.length + 1), text: readFileSync(path, 'utf8') }
    }
  }
}

test('collectTokens finds definitions in both stylesheet and inline-binding form', () => {
  const usage = collectTokens([
    { path: 'a.css', text: ':root { --bg-base: #fff; }\n.x { color: var(--bg-base); }' },
    { path: 'b.vue', text: "style: { '--sidebar-width': w }" },
  ])
  assert.equal(usage.defined.has('--bg-base'), true)
  assert.equal(usage.defined.has('--sidebar-width'), true)
  assert.deepEqual(usage.referenced.get('--bg-base'), ['a.css:2'])
})

test('undefinedTokens reports a reference with no definition, and where it is', () => {
  const usage = collectTokens([{ path: 'a.css', text: '.x { color: var(--nope, #fff); }' }])
  assert.deepEqual([...undefinedTokens(usage)], [['--nope', ['a.css:1']]])
})

test('undefinedTokens sees through a fallback into a nested reference', () => {
  // `var(--defined, var(--typo))` never evaluates the inner name, so the typo
  // is invisible in the rendered page — but it is still a dangling reference.
  const usage = collectTokens([
    { path: 'a.css', text: ':root { --real: 1px; }\n.x { border: var(--real, var(--typo)); }' },
  ])
  assert.deepEqual([...undefinedTokens(usage).keys()], ['--typo'])
})

test('every CSS variable the frontend references is defined somewhere', () => {
  // The regression guard. A `var(--typo)` fails silently — with a fallback it
  // uses a hardcoded literal (theme-blind, so wrong in one theme), without one
  // the declaration is dropped. Nothing else in the toolchain catches it.
  const usage = collectTokens(sources(SRC, DEFINITION_FILES))
  const missing = undefinedTokens({
    defined: usage.defined,
    referenced: collectTokens(sources(SRC, REFERENCE_FILES)).referenced,
  })
  const report = [...missing]
    .map(([name, sites]) => `  ${name}\n    ${sites.join('\n    ')}`)
    .join('\n')
  assert.equal(missing.size, 0, `undefined CSS variables:\n${report}`)
})
