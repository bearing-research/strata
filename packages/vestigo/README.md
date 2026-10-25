# vestigo

<!-- --8<-- [start:body] -->
The ledger of the shared computation substrate, factored out of Strata's
artifact store (ADR 002): immutable versioned artifacts deduplicated by
provenance hash, builds with leases and fenced attempts, mutable references
with compare-and-set and approval, retention and race-safe garbage
collection, all in one transaction discipline on SQLite or PostgreSQL, with
blob backends for local disk, S3, GCS and Azure.

Strata is its first consumer, and until ADR 002 E2 lands the package is not
usable on its own: garbage collection, verification and the S3 settings still
import Strata. *Vestigo* is Latin for "I track down, I trace":
the ledger traces every artifact back to the computation and inputs that made it.
<!-- --8<-- [end:body] -->

## Formal models

`formal/tla/` holds the TLA+ models of the two lifecycles this package
implements, `ArtifactLifecycle.tla` and `BuildLease.tla`, with the
configurations that found and then closed their bugs. See the repository's
`formal/README.md` for how to run them.
