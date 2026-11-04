/** A note's first line of text. A heading at any level is its title, not its preview. */
export function notePreview(source: string): string {
  const lines = source.split('\n').map((line) => line.trim())
  return lines.find((line) => line && !/^#+\s/.test(line)) ?? ''
}
