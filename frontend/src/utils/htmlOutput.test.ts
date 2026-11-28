import { describe, expect, it } from 'vitest'
import { MAX_HTML_OUTPUT_HEIGHT, reportedHeight } from './htmlOutput'

const frame = {} as Window
const other = {} as Window

describe('reportedHeight', () => {
  it('takes the height its own frame reports, rounded up', () => {
    expect(reportedHeight({ source: frame, data: { strataOutputHeight: 466.2 } }, frame)).toBe(467)
  })

  it('ignores any other window, such as another output or a popup', () => {
    expect(reportedHeight({ source: other, data: { strataOutputHeight: 300 } }, frame)).toBeNull()
  })

  it('ignores a message that is not a usable height', () => {
    for (const data of [null, 'x', {}, { strataOutputHeight: '300' }, { strataOutputHeight: 0 }]) {
      expect(reportedHeight({ source: frame, data }, frame)).toBeNull()
    }
    expect(
      reportedHeight({ source: frame, data: { strataOutputHeight: Infinity } }, frame),
    ).toBeNull()
  })

  it('caps a page that reports itself taller than any output should be', () => {
    expect(reportedHeight({ source: frame, data: { strataOutputHeight: 1e9 } }, frame)).toBe(
      MAX_HTML_OUTPUT_HEIGHT,
    )
  })

  it('has nothing to compare with before the frame exists', () => {
    expect(reportedHeight({ source: frame, data: { strataOutputHeight: 300 } }, null)).toBeNull()
  })
})
