# Worker Pool

!!! info "This page is the `strata-pool` package README"

    It is included verbatim from `packages/strata-pool/README.md`, which is
    also the package's PyPI landing page. One source, so the two cannot drift.

    The pool is a **separate package** — `pip install strata-pool`.

    It sits at a **different layer** from the workers a notebook dispatches to:
    nothing in `strata` imports it, and `# @worker` cannot name a pool machine.
    A proxy above both composes them, checking the artifact store first and
    submitting to the pool only on a miss. Machines run the same
    `strata-worker` binary either way.

    If what you want is hardware you keep running and point cells at, see
    [Distributed Workers](workers.md#worker-pools-a-different-layer), which
    compares the two side by side.

--8<-- "packages/strata-pool/README.md:body"
