# vestigo

<!-- --8<-- [start:body] -->
The ledger of the shared computation substrate, factored out of Strata's
artifact store (ADR 002): immutable versioned artifacts deduplicated by
provenance hash, builds with leases and fenced attempts, mutable references
with compare-and-set and approval, retention and race-safe garbage
collection, all in one transaction discipline on SQLite or PostgreSQL, with
blob backends for local disk, S3, GCS and Azure.

Strata is its first consumer. The package imports nothing of Strata: a
consumer passes blob settings in (`BlobSettings`), registers the validators
`verify_artifacts` runs per content type (`register_validator`), and records
what a version holds (`retain`) or requires (`require`, `requires=` on
`create_artifact`), which garbage collection protects as one closure. A
structural graph of content-addressed nodes (`graph`) records what produced
a version (`definition_id`) and is walked with bounded traversals; the
ledger never parses a node's payload. A build's releases (`releases`) make
its outputs current together, any number of checkpoints and one final, each
fenced on the attempt's lease and accepted only once every output's bytes
are stored, and a build is found again by its computation
(`find_open_build`) for the attempt that resumes it; the typed operations a release carries (`operations`) move
references and assign authorities in the same transaction.
*Vestigo* is Latin for "I track down, I trace": the ledger traces every
artifact back to the computation and inputs that made it.
<!-- --8<-- [end:body] -->

## Formal models

`formal/tla/` holds the TLA+ models of the two lifecycles this package
implements, `ArtifactLifecycle.tla` and `BuildLease.tla`, with the
configurations that found and then closed their bugs. `formal/README.md`
says how to run them.

Artifact and name URIs are ``{scheme}://artifact/{id}@v={n}`` and
``{scheme}://name/{name}``; the scheme is ``vestigo`` unless the consumer calls
``vestigo.uris.set_uri_scheme`` once at import (Strata uses ``strata``). The
scheme is persisted in ``input_versions`` rows, so a store keeps the scheme it
was written under.

## Install

```bash
uv add vestigo            # or: pip install vestigo
uv add "vestigo[postgres]"  # PostgreSQL metadata; also [s3], [gcs], [azure] for blobs
```

## Development

```bash
uv sync --all-extras      # the package, every extra and the dev group
uv run pytest             # the suite; the Postgres tests start a container through Docker
uv run pre-commit run --all-files
```

Strata's `tests/contract/` holds the cross-repository compatibility tests:
hash vectors for the graph encoders, operation schemas, output keys, and the
ledger's table inventory. A change to any of those is a contract change and
needs a matching Strata release.
