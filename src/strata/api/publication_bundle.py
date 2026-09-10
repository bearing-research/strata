"""Build the self-contained bundle a publication is archived as.

A hosted link resolves for as long as the server does, and a URL printed in a
paper outlives most servers. The bundle is the copy that needs neither: a page,
the bytes, a machine-readable record, an RO-Crate a repository can ingest, and
a README naming the digest.

It is written once here because two callers want it — ``strata artifact
archive``, which opens the store directly, and ``GET /p/{token}/archive.zip``,
which is how a service holding only HTTP access to the store gets the same
thing. Two implementations of a set of files that describe each other would
drift, and the drift would be silent: both would keep producing a bundle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore, ArtifactVersion, Publication

# Extension by content type, for naming the file inside a bundle. A reader
# should be able to double-click it; ``.bin`` helps nobody.
BUNDLE_EXTENSIONS = {
    "image/png": ".png",
    "text/markdown": ".md",
    "json/object": ".json",
    "arrow/ipc": ".arrow",
    "pickle/object": ".pickle",
}


def write_bundle(
    store: ArtifactStore,
    artifact: ArtifactVersion,
    dest: Path,
    *,
    publication: Publication,
    max_depth: int = 10,
    tenant: str | None = None,
) -> list[str]:
    """Write the bundle into *dest*, and return its filenames in reading order.

    *dest* must already exist and is not emptied: whether writing into an
    occupied directory is a mistake is the caller's question, and the CLI and
    the route answer it differently — one is a person naming a path, the other
    a temporary directory nobody else can see.

    Raises:
        ValueError: If the artifact is not readable or has no stored bytes. A
            half-written blob has a digest like any other, so without this an
            artifact still ``building`` archives cleanly and the bundle
            presents truncated bytes as a deposit-ready record, with a
            sha256sum line vouching for the fragment.
    """
    from strata.api.provenance_ld import build_crate
    from strata.api.publication_page import build_record, content_type_of, render_publication
    from strata.services.artifact import ArtifactService

    if artifact.state not in ("ready", "superseded"):
        raise ValueError(
            f"{artifact.id}@v={artifact.version} is not readable (state={artifact.state})"
        )

    digest = publication.content_sha256 or store.content_digest(artifact.id, artifact.version)
    if digest is None:
        raise ValueError(f"{artifact.id}@v={artifact.version} has no stored bytes to archive")

    content_type = content_type_of(artifact)
    filename = f"artifact{BUNDLE_EXTENSIONS.get(content_type, '.bin')}"

    lineage = ArtifactService().build_lineage(
        store,
        artifact=artifact,
        artifact_id=artifact.id,
        version=artifact.version,
        tenant_filter=tenant,
        max_depth=max_depth,
    )

    reader_cm = store.open_blob_reader(artifact.id, artifact.version)
    if reader_cm is None:
        raise ValueError(f"{artifact.id}@v={artifact.version} has no stored bytes to archive")
    with reader_cm as reader, open(dest / filename, "wb") as out:
        while chunk := reader.read(1024 * 1024):
            out.write(chunk)

    # A relative reference, not a data URI: the file is right there, so
    # embedding it would double the bundle's size for nothing and turn a large
    # figure into an index.html no browser will open — the one thing a bundle
    # has to guarantee. (The hosted page inlines because it has no such file.)
    image_src = filename if content_type == "image/png" else None

    (dest / "index.html").write_text(
        render_publication(
            publication=publication,
            artifact=artifact,
            lineage=lineage,
            content_type=content_type,
            image_src=image_src,
            bundle_filename=filename,
        ),
        encoding="utf-8",
    )
    (dest / "manifest.json").write_text(
        json.dumps(
            build_record(
                publication=publication,
                artifact=artifact,
                lineage=lineage,
                content_type=content_type,
                archived=True,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    # RO-Crate is what a repository ingests. Without it a Zenodo deposit is a
    # folder a human can read; with it the chain is data the archive can index.
    (dest / "ro-crate-metadata.json").write_text(
        json.dumps(
            build_crate(
                publication=publication,
                artifact=artifact,
                lineage=lineage,
                content_type=content_type,
                payload_id=filename,
                include_descriptor=True,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    parquet_name = _write_parquet_companion(store, artifact, dest)
    _name_bundle_files(dest, filename, parquet_name)

    (dest / "README.md").write_text(
        _bundle_readme(publication, artifact, filename, digest, parquet_name),
        encoding="utf-8",
    )

    written = ["index.html", filename]
    if parquet_name is not None:
        written.append(parquet_name)
    written += ["manifest.json", "ro-crate-metadata.json", "README.md"]
    return written


def _write_parquet_companion(
    store: ArtifactStore, artifact: ArtifactVersion, dest: Path
) -> str | None:
    """Write ``artifact.parquet`` beside the Arrow bytes, for tabular artifacts.

    Arrow IPC is a transport format. It has a stability promise, but a data
    repository indexes Parquet and a reader in a decade will reach for it with
    whatever tool they have. The Arrow file stays: it is the archived bytes and
    the digest in the manifest covers it. This is a second, more portable
    rendering of the same rows.

    Returns the filename, or ``None`` when the artifact is not tabular — an
    image or a pickle has no rows to write and gets no companion.
    """
    from strata.notebook.serializer import write_table_export

    reader_cm = store.open_blob_reader(artifact.id, artifact.version)
    if reader_cm is None:
        return None
    with reader_cm as reader:
        blob = reader.read()

    parquet = write_table_export(blob, "parquet")
    if parquet is None:
        return None
    (dest / "artifact.parquet").write_bytes(parquet)
    return "artifact.parquet"


def _name_bundle_files(dest: Path, filename: str, parquet_name: str | None) -> None:
    """Say which file each digest covers.

    ``content_sha256`` was unambiguous while a bundle held one payload: there
    was only one thing it could describe. A second file makes it a claim about
    an unnamed file, and a digest that does not say what it covers is worse
    than none in a bundle meant to be read long after anyone is left to ask.

    So the record names its payload, and any companion carries its own digest
    beside it.
    """
    manifest_path = dest / "manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["content_file"] = filename
    if parquet_name is not None:
        record["additional_files"] = [
            {
                "file": parquet_name,
                "content_type": "application/vnd.apache.parquet",
                "sha256": hashlib.sha256((dest / parquet_name).read_bytes()).hexdigest(),
                "note": "The same rows as the archived bytes, in Parquet.",
            }
        ]
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _bundle_readme(
    publication: Any,
    artifact: ArtifactVersion,
    filename: str,
    digest: str,
    parquet_name: str | None = None,
) -> str:
    title = publication.title or f"{artifact.id}@v={artifact.version}"
    parquet_line = (
        f"\n- `{parquet_name}` — the same rows in Parquet, for tools that read it."
        if parquet_name
        else ""
    )
    parquet_note = (
        f" `{parquet_name}` is a rendering of the same rows and has its own"
        " digest in `manifest.json`."
        if parquet_name
        else ""
    )
    return f"""# {title}

A Strata artifact and the record of what produced it.

- `index.html` — the result, the code that produced it, and the code and
  environment of every step behind it. Open it in a browser; it needs no
  server and makes no external requests.
- `{filename}` — the bytes themselves.{parquet_line}
- `manifest.json` — the same record, machine-readable.
- `ro-crate-metadata.json` — the same chain as [RO-Crate](https://w3id.org/ro/crate/)
  JSON-LD, which repositories and provenance tooling read directly.

## Checking it

The archived bytes are byte-identical to what was archived if:

```
sha256sum {filename}
# {digest}
```

`{filename}` is what that digest covers.{parquet_note}

That is the whole of what this bundle can prove about the contents. It shows
nothing about whether the result was honestly produced — no digest could — and
it does not claim the computation was reproduced. Re-running it is a separate
matter, and one only you can do: the source and environment in `index.html`
are what it would take.
"""
