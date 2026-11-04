// How a notebook's stored outputs read. Beside a cell's variables its store holds records of
// what it displayed, printed and wrote to an output folder, under reserved names.
export function outputLabel(output: string | null | undefined, artifactId: string): string {
  if (!output) return artifactId
  const display = /^__display__(\d+)/.exec(output)
  if (display) return `display ${display[1]}`
  if (output.startsWith('__console__')) return 'console'
  if (output.startsWith('__file__')) return 'file'
  return output
}
