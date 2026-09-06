# Worker Pool

!!! info "This page is the `strata-pool` package README"

    It is included verbatim from `packages/strata-pool/README.md`, which is
    also the package's PyPI landing page. One source, so the two cannot drift.

    The pool is a **separate package** — `pip install strata-pool`. The
    notebook never imports it; a proxy composes the two. For long-lived
    workers you start and manage yourself, see
    [Distributed Workers](workers.md) instead.

--8<-- "packages/strata-pool/README.md:body"
