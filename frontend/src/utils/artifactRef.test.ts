import assert from 'node:assert/strict'
import test from 'node:test'

import { parseArtifactRef, parseArtifactUris } from './artifactRef.ts'

test('a notebook artifact ref splits into id and version', () => {
  assert.deepEqual(parseArtifactRef('strata://artifact/nb_x_cell_c1_var_model@v=3'), {
    id: 'nb_x_cell_c1_var_model',
    version: 3,
  })
})

test('the version is the last @v=, so an id containing one survives', () => {
  assert.deepEqual(parseArtifactRef('strata://artifact/a@v=1@v=2'), { id: 'a@v=1', version: 2 })
})

test('anything that is not an artifact ref is refused', () => {
  assert.equal(parseArtifactRef('strata://table/x'), null)
  assert.equal(parseArtifactRef('strata://artifact/noversion'), null)
  assert.equal(parseArtifactRef('strata://artifact/@v=1'), null)
  assert.equal(parseArtifactRef('strata://artifact/x@v=zero'), null)
})

test('a uri map keeps only real refs', () => {
  assert.deepEqual(parseArtifactUris({ model: 'strata://artifact/m@v=1', junk: 7, bad: 'nope' }), {
    model: 'strata://artifact/m@v=1',
  })
  assert.deepEqual(parseArtifactUris(null), {})
})
