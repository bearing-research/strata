// Where the Strata server is. Behind a reverse proxy at a non-root path the server injects
// <meta name="strata-base-path">, and every API and WebSocket URL has to carry that path.

export function injectedBasePath(doc: Pick<Document, 'querySelector'> | undefined): string {
  const content = doc?.querySelector('meta[name="strata-base-path"]')?.getAttribute('content')
  return (content ?? '').replace(/\/+$/, '')
}

export function strataHttpBase(
  configured: string | undefined,
  origin: string | undefined,
  basePath: string,
): string {
  if (configured) return configured
  return `${origin ?? 'http://localhost:8765'}${basePath}`
}

export function strataWsBase(httpBase: string): string {
  if (httpBase.startsWith('https://')) {
    return `wss://${httpBase.slice('https://'.length)}`
  }
  if (httpBase.startsWith('http://')) {
    return `ws://${httpBase.slice('http://'.length)}`
  }
  return httpBase.replace(/^http/, 'ws')
}

// This page's base path, read once: the server injects it into index.html.
export const BASE_PATH = injectedBasePath(typeof document !== 'undefined' ? document : undefined)
