// Beside the version picker, which Material adds to this same element once versions.json loads.
const topic = document.querySelector(".md-header__topic");
if (topic && typeof __md_scope !== "undefined") {
  const link = document.createElement("a");
  link.className = "changelog-link";
  link.href = new URL("changelog/", __md_scope);
  link.textContent = "Changelog";
  topic.append(link);
}
