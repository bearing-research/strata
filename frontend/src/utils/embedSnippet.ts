// Export menu "Embed": an app-view iframe snippet plus its auto-resize listener.
// The listener takes heights only from the Strata origin, so another frame on the
// host page cannot resize the notebook's iframe.
export function embedSnippet(origin: string, sessionId: string): string {
  const url = `${origin}/#/app/${sessionId}?embed=1`
  return [
    `<iframe src="${url}" title="Strata notebook" style="width:100%;border:0"></iframe>`,
    `<script>addEventListener('message',e=>{if(e.origin===${JSON.stringify(origin)}&&e.data&&e.data.type==='strata:embed:resize')`,
    `document.querySelector('iframe[title=\\'Strata notebook\\']').style.height=e.data.height+'px'})<\/script>`,
  ].join('\n')
}
