// The "Copy page as Markdown" button. docs_hooks/llms_txt.py publishes each page's source as
// index.md beside its index.html.
document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy-markdown]");
  if (!button) return;
  const url = new URL("index.md", window.location.href);
  const text = fetch(url).then(async (response) => {
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    return new Blob([await response.text()], { type: "text/plain" });
  });
  const title = button.title;
  try {
    // A promise here, because Safari drops the click's permission across an await.
    await navigator.clipboard.write([
      new ClipboardItem({ "text/plain": text }),
    ]);
    button.title = "Copied";
    button.classList.add("copied");
  } catch {
    // No clipboard here (plain http, a denied permission): the page itself is the next best.
    window.open(url, "_blank", "noopener");
    return;
  }
  setTimeout(() => {
    button.title = title;
    button.classList.remove("copied");
  }, 1500);
});
