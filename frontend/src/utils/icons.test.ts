import assert from 'node:assert/strict'
import { readdirSync, readFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import test from 'node:test'

const SRC = new URL('..', import.meta.url).pathname

// Arrows in a sentence and the lineage tree's rule are text, and every font has them.
const ALLOWED = new Set(['→', '└', '─'])
const GLYPHS = /[←-⇿⌀-⏿①-➿⤀-⧿⬀-⯿️\u{1f000}-\u{1faff}]/gu
const ENTITIES = /&#(x[0-9a-f]+|\d+);|&times;|&rarr;/gi
const ESCAPES = /\\u(25|26|27|21|23)[0-9a-f]{2}/gi

function sources(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) return sources(path)
    return /\.(vue|ts)$/.test(entry.name) && !entry.name.endsWith('.test.ts') ? [path] : []
  })
}

function withoutComments(text: string): string {
  return text
    .split(/<!--[\s\S]*?-->/)
    .join('')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '')
    .replace(/\s\/\/ .*$/gm, '')
}

test('the UI draws no icon with a text glyph', () => {
  const found: string[] = []
  for (const path of sources(SRC)) {
    const text = withoutComments(readFileSync(path, 'utf8'))
    const glyphs = [...text.matchAll(GLYPHS)].map((m) => m[0]).filter((g) => !ALLOWED.has(g))
    const encoded = [...text.matchAll(ENTITIES), ...text.matchAll(ESCAPES)]
      .map((m) => m[0])
      // A brace escaped for a template, and a newline in a placeholder.
      .filter((e) => !['&#123;', '&#125;', '&#10;', '&rarr;'].includes(e))
    if (glyphs.length || encoded.length) {
      found.push(`${relative(SRC, path)}: ${[...glyphs, ...encoded].join(' ')}`)
    }
  }
  assert.deepEqual(found, [])
})

test('a button with only an icon has a name', () => {
  const found: string[] = []
  for (const path of sources(SRC).filter((p) => p.endsWith('.vue'))) {
    const text = readFileSync(path, 'utf8')
    const template = text.slice(text.indexOf('<template>'))
    for (const [button, attrs, body] of template.matchAll(
      /<button\b((?:[^>"]|"[^"]*")*)>([\s\S]*?)<\/button>/g,
    )) {
      const withoutTags = body
        .split(/<[^>]*>/)
        .join('')
        .replace(/\{\{[\s\S]*?\}\}/g, 'text')
        .trim()
      if (!withoutTags && !/aria-label/.test(attrs)) {
        found.push(`${relative(SRC, path)}: ${button.replace(/\s+/g, ' ').slice(0, 80)}`)
      }
    }
  }
  assert.deepEqual(found, [])
})

test('the tab icon is linked, and is the same diamond as the logo', () => {
  const frontend = new URL('../../', import.meta.url).pathname
  assert.match(readFileSync(join(frontend, 'index.html'), 'utf8'), /rel="icon"[^>]*\/favicon\.svg/)
  const diamond = readFileSync(
    join(frontend, 'node_modules/@lucide/vue/dist/esm/icons/diamond.mjs'),
    'utf8',
  ).match(/d: "([^"]+)"/)?.[1]
  assert.ok(diamond)
  assert.ok(readFileSync(join(frontend, 'public/favicon.svg'), 'utf8').includes(diamond))
})
