import assert from 'node:assert/strict'
import { test } from 'node:test'
import { isLive, withLive } from './widgetLive.ts'

test('turning live on puts the line first', () => {
  assert.equal(withLive('a = slider(0, 1)', true), '# @live\na = slider(0, 1)')
})

test('turning live on replaces an off line rather than sitting before it', () => {
  // The last `@live` in the block wins, so a leftover `off` would keep it off.
  assert.equal(withLive('# @live off\na = slider(0, 1)', true), '# @live\na = slider(0, 1)')
})

test('turning live off removes the line from the leading block only', () => {
  const source = '# @name controls\n# @live\na = slider(0, 1)\n# @live'
  assert.equal(withLive(source, false), '# @name controls\na = slider(0, 1)\n# @live')
})

test('live is read from the leading block, the last line winning', () => {
  assert.equal(isLive('# @live\na = slider(0, 1)'), true)
  assert.equal(isLive('# @live\n# @live off\na = slider(0, 1)'), false)
  assert.equal(isLive('# @live FALSE\na = slider(0, 1)'), false)
  // Below the code nothing reads it.
  assert.equal(isLive('a = slider(0, 1)\n# @live'), false)
})

test('a toggle turns it the other way, twice back where it was', () => {
  const source = 'a = slider(0, 1)'
  const once = withLive(source, !isLive(source))
  assert.equal(isLive(once), true)
  assert.equal(withLive(once, !isLive(once)), source)
})
