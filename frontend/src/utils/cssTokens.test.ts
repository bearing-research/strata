import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

import { collectTokens, undefinedTokens } from './cssTokens.ts'

const SRC = join(fileURLToPath(new URL('.', import.meta.url)), '..')
// A .ts file only defines variables (inline style bindings). Tests are skipped:
// their fixtures use deliberately undefined names.
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
  // The inner name never evaluates, so the typo is invisible on the page.
  const usage = collectTokens([
    { path: 'a.css', text: ':root { --real: 1px; }\n.x { border: var(--real, var(--typo)); }' },
  ])
  assert.deepEqual([...undefinedTokens(usage).keys()], ['--typo'])
})

test('a reference is found even when the token name wraps to the next line', () => {
  // Collected per line, this reference would be missed.
  const usage = collectTokens([
    { path: 'a.css', text: '.x {\n  color: var(\n    --wrapped\n  );\n}' },
  ])
  assert.deepEqual([...undefinedTokens(usage).keys()], ['--wrapped'])
})

test('prose in a comment is not a definition', () => {
  // A comment counted as a declaration would let a typo pass if it is mentioned.
  const usage = collectTokens([
    { path: 'a.css', text: '/* --documented: what it does */\n.x { color: var(--documented); }' },
  ])
  assert.deepEqual([...undefinedTokens(usage).keys()], ['--documented'])
})

test('blanking a comment keeps later line numbers honest', () => {
  const usage = collectTokens([
    { path: 'a.css', text: '/* a\n   multi-line\n   comment */\n.x { color: var(--nope); }' },
  ])
  assert.deepEqual(usage.referenced.get('--nope'), ['a.css:4'])
})

test('every CSS variable the frontend references is defined somewhere', () => {
  // The real check: a `var(--typo)` fails silently and nothing else catches it.
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
