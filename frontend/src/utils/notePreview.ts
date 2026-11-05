import MarkdownIt from 'markdown-it'

const md = new MarkdownIt({ html: false })

type Token = ReturnType<typeof md.parse>[number]

/** An inline token's first line, with its markup gone. */
function plainFirstLine(inline: Token): string {
  let text = ''
  for (const child of inline.children ?? []) {
    if (child.type === 'softbreak' || child.type === 'hardbreak') break
    // An image's content is its alt text.
    if (['text', 'code_inline', 'image'].includes(child.type)) text += child.content
  }
  return text.trim()
}

function firstLine(token: Token): string {
  if (token.type === 'inline') return plainFirstLine(token)
  if (token.type === 'fence' || token.type === 'code_block') {
    return token.content.split('\n')[0].trim()
  }
  return ''
}

/** A note's first heading as plain text, at any level. */
export function noteHeading(source: string): string | null {
  const tokens = md.parse(source, {})
  const open = tokens.findIndex((token) => token.type === 'heading_open')
  return open < 0 ? null : plainFirstLine(tokens[open + 1]) || null
}

/** A note's first line of text, without markup. A heading is its title, not its preview. */
export function notePreview(source: string): string {
  const tokens = md.parse(source, {})
  for (const [i, token] of tokens.entries()) {
    if (tokens[i - 1]?.type === 'heading_open') continue
    const line = firstLine(token)
    if (line) return line
  }
  return ''
}
