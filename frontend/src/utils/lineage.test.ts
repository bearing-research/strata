import assert from 'node:assert/strict'
import test from 'node:test'

import type { LineageGraph } from '../composables/useStrata.ts'
import { flattenLineage, lineageToTree } from './lineage.ts'

// model ← features ← scan ← table@snapshot. Edges run input → consumer.
const MODEL = 'strata://artifact/m@v=3'
const FEATURES = 'strata://artifact/f@v=1'
const SCAN = 'strata://artifact/s@v=1'
const TABLE = 'file:///wh#nyc.trips@snapshot=42'

const graph: LineageGraph = {
  artifact_uri: MODEL,
  nodes: [
    { uri: MODEL, type: 'artifact', version: 3, transform_ref: 'train_hgbr@v1' },
    { uri: FEATURES, type: 'artifact', version: 1, transform_ref: 'feature_eng@v1' },
    { uri: SCAN, type: 'artifact', version: 1, transform_ref: 'scan@v1' },
    { uri: TABLE, type: 'table' },
  ],
  edges: [
    { from_uri: FEATURES, to_uri: MODEL },
    { from_uri: SCAN, to_uri: FEATURES },
    { from_uri: TABLE, to_uri: SCAN },
  ],
}

test('lineageToTree builds the chain rooted at the artifact', () => {
  const tree = lineageToTree(graph, MODEL)
  assert.equal(tree.label, 'train_hgbr@v1')
  assert.equal(tree.version, 3)
  assert.equal(tree.children.length, 1)
  assert.equal(tree.children[0].label, 'feature_eng@v1')
  assert.equal(tree.children[0].children[0].label, 'scan@v1')
  const table = tree.children[0].children[0].children[0]
  assert.equal(table.type, 'table')
  assert.equal(table.label, 'nyc.trips @ snapshot 42')
})

test('flattenLineage yields indented rows in chain order', () => {
  const rows = flattenLineage(lineageToTree(graph, MODEL))
  assert.deepEqual(
    rows.map((r) => [r.depth, r.label]),
    [
      [0, 'train_hgbr@v1'],
      [1, 'feature_eng@v1'],
      [2, 'scan@v1'],
      [3, 'nyc.trips @ snapshot 42'],
    ],
  )
})

test('lineageToTree tolerates cycles without looping forever', () => {
  const cyclic: LineageGraph = {
    artifact_uri: MODEL,
    nodes: [{ uri: MODEL, type: 'artifact', version: 1, transform_ref: 't@v1' }],
    edges: [{ from_uri: MODEL, to_uri: MODEL }],
  }
  const rows = flattenLineage(lineageToTree(cyclic, MODEL))
  assert.equal(rows.length, 1)
})
