"""Build the self-contained bundle a publication is archived as.

A page, the bytes, a machine-readable record, an RO-Crate and a README naming the
digest, needing no server. ``strata artifact archive`` and
``GET /p/{token}/archive.zip`` share this one implementation so they cannot drift.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import uuid
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


def payload_filename(content_type: str) -> str:
    """The name the published bytes take inside a bundle."""
    return f"artifact{BUNDLE_EXTENSIONS.get(content_type, '.bin')}"


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
    filename = payload_filename(content_type)

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
    parquet_name = _write_parquet_companion(store, artifact, dest)
    companions = (
        [
            {
                "file": parquet_name,
                "content_type": PARQUET_CONTENT_TYPE,
                "sha256": hashlib.sha256((dest / parquet_name).read_bytes()).hexdigest(),
                "note": "The same rows as the archived bytes, in Parquet.",
            }
        ]
        if parquet_name is not None
        else []
    )
    _name_bundle_files(dest, filename, companions)
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
                companions=companions,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    (dest / "README.md").write_text(
        _bundle_readme(publication, artifact, filename, digest, parquet_name),
        encoding="utf-8",
    )

    written = ["index.html", filename]
    if parquet_name is not None:
        written.append(parquet_name)
    written += ["manifest.json", "ro-crate-metadata.json", "README.md"]
    return written


PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"

# A tabular artifact larger than this is archived without its Parquet copy. Fixed, so a
# publication's archive stays the same bytes on every build.
PARQUET_COMPANION_MAX_BYTES = 128 * 1024 * 1024

# How far a publication's page, record, card, crate and archive walk its chain: one depth, so
# the archive never shows fewer steps than the page.
PUBLICATION_MAX_DEPTH = 25

# Under the store's directory: each publication's built archive, in a directory per token.
ARCHIVE_CACHE_DIRNAME = "publication-archives"

# Every zip member gets this timestamp and these permissions, so two archives of
# one publication are byte-identical (``ZipFile.write`` stamps each mtime).
_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = 0o644 << 16


def bundle_zip(
    store: ArtifactStore,
    artifact: ArtifactVersion,
    out: Path,
    *,
    publication: Publication,
    max_depth: int = 10,
    tenant: str | None = None,
) -> str:
    """Write the bundle to *out* as one zip, byte for byte the same each time it is built.

    Members are in :func:`write_bundle`'s reading order, streamed from disk so an
    artifact's size never sits in memory. Returns the zip's sha256 hex digest.

    Raises:
        ValueError: As :func:`write_bundle`.
    """
    with tempfile.TemporaryDirectory() as workdir:
        dest = Path(workdir)
        written = write_bundle(
            store, artifact, dest, publication=publication, max_depth=max_depth, tenant=tenant
        )
        with (
            open(out, "wb") as handle,
            zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as bundle,
        ):
            for name in written:
                info = zipfile.ZipInfo(name, date_time=_ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = _ZIP_FILE_MODE
                # Known up front, as ``writestr`` would set it, so the headers (and the
                # zip64 decision) match an in-memory build exactly.
                info.file_size = (dest / name).stat().st_size
                with bundle.open(info, "w") as member, open(dest / name, "rb") as source:
                    shutil.copyfileobj(source, member, 1024 * 1024)
    digest = hashlib.sha256()
    with open(out, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cached_bundle_zip(
    store: ArtifactStore, artifact: ArtifactVersion, *, publication: Publication
) -> tuple[Path, str]:
    """The zip :func:`bundle_zip` builds for *publication*, built once and kept on disk.

    Keyed by everything in the publication record, so editing its credits rebuilds it.
    Returns ``(path, sha256 hex digest)``.

    Raises:
        ValueError: As :func:`write_bundle`.
    """
    record = {k: v for k, v in dataclasses.asdict(publication).items() if k != "revoked_at"}
    key = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
    cache_dir = _archive_cache_dir(store, publication.token)
    prefix = f"{key[:32]}-"
    cached = next(cache_dir.glob(f"{prefix}*.zip"), None)
    if cached is not None:
        return cached, cached.stem.removeprefix(prefix)

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cache_dir / f"{prefix}{uuid.uuid4().hex}.partial"
    try:
        digest = bundle_zip(
            store, artifact, partial, publication=publication, max_depth=PUBLICATION_MAX_DEPTH
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    final = cache_dir / f"{prefix}{digest}.zip"
    os.replace(partial, final)
    # An archive of an earlier record of this publication is never served again.
    for stale in cache_dir.glob("*.zip"):
        if stale != final:
            stale.unlink(missing_ok=True)
    return final, digest


def drop_cached_bundles(store: ArtifactStore, token: str) -> None:
    """Remove every cached archive of the publication *token*."""
    cache_dir = _archive_cache_dir(store, token)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _archive_cache_dir(store: ArtifactStore, token: str) -> Path:
    return store.artifact_dir / ARCHIVE_CACHE_DIRNAME / token


def _write_parquet_companion(
    store: ArtifactStore, artifact: ArtifactVersion, dest: Path
) -> str | None:
    """Write ``artifact.parquet`` beside the Arrow bytes, for tabular artifacts.

    Repositories index Parquet; the Arrow file stays as the archived, digested
    bytes. Returns the filename, or ``None`` when the artifact is not tabular or is
    larger than :data:`PARQUET_COMPANION_MAX_BYTES`.
    """
    from strata.notebook.serializer import write_table_export

    # The conversion holds the whole table in memory; past this the Arrow file stands alone.
    if (store.blob_size(artifact.id, artifact.version) or 0) > PARQUET_COMPANION_MAX_BYTES:
        return None

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


def _name_bundle_files(dest: Path, filename: str, companions: list[dict[str, str]]) -> None:
    """Name the payload file in ``manifest.json`` and give any companion its own digest.

    With more than one file, a digest that does not say what it covers is ambiguous.
    """
    manifest_path = dest / "manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["content_file"] = filename
    if companions:
        record["additional_files"] = companions
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def companion_digests(archive: Path) -> dict[str, str]:
    """``{sha256: filename}`` for the companion files listed in a built archive's manifest."""
    with zipfile.ZipFile(archive) as bundle:
        record = json.loads(bundle.read("manifest.json"))
    return {entry["sha256"]: entry["file"] for entry in record.get("additional_files", [])}


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
