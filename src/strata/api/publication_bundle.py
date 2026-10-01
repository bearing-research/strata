"""Build the self-contained bundle a publication is archived as.

A page, the bytes, a machine-readable record, an RO-Crate and a README naming the
digest, needing no server. ``strata artifact archive`` and
``GET /p/{token}/archive.zip`` share this one implementation so they cannot drift.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore, ArtifactVersion, Publication

# Extension by content type for the file inside a bundle; ``.bin`` helps nobody.
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

    *dest* must already exist and is not emptied; the caller decides whether an
    occupied directory is a mistake.

    Raises:
        ValueError: If the artifact is not ready/superseded or has no stored bytes
            (a still-``building`` blob would otherwise archive truncated bytes).
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

    # A relative reference, not a data URI: embedding doubles the bundle and can
    # make index.html too large for a browser to open. (The hosted page inlines
    # because it has no such file.)
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
    # RO-Crate turns the chain into data a repository such as Zenodo can index.
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


# Every zip member gets this timestamp and these permissions, so two archives of
# one publication are byte-identical (``ZipFile.write`` stamps each mtime).
_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = 0o644 << 16


def bundle_zip(
    store: ArtifactStore,
    artifact: ArtifactVersion,
    *,
    publication: Publication,
    max_depth: int = 10,
    tenant: str | None = None,
) -> bytes:
    """The bundle as one zip, byte for byte the same each time it is built.

    Members are in :func:`write_bundle`'s reading order.

    Raises:
        ValueError: As :func:`write_bundle`.
    """
    with tempfile.TemporaryDirectory() as workdir:
        dest = Path(workdir)
        written = write_bundle(
            store, artifact, dest, publication=publication, max_depth=max_depth, tenant=tenant
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name in written:
                info = zipfile.ZipInfo(name, date_time=_ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = _ZIP_FILE_MODE
                bundle.writestr(info, (dest / name).read_bytes())
    return buffer.getvalue()


def _write_parquet_companion(
    store: ArtifactStore, artifact: ArtifactVersion, dest: Path
) -> str | None:
    """Write ``artifact.parquet`` beside the Arrow bytes, for tabular artifacts.

    Repositories index Parquet; the Arrow file stays as the archived, digested
    bytes. Returns the filename, or ``None`` when the artifact is not tabular.
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
    """Name the payload file in ``manifest.json`` and give any companion its own digest.

    With more than one file, a digest that does not say what it covers is ambiguous.
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
