// A cell's console after one `cell_console` frame. Chunk 0 of a remote run's
// stream starts it afresh, so the last run's text does not stay above the new
// run's; every other frame appends.
export function applyConsoleChunk(
  existing: string | undefined,
  text: string,
  chunkSeq: unknown,
): string {
  return (chunkSeq === 0 ? '' : (existing ?? '')) + text
}
