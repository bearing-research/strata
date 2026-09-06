# Publishing an artifact

A figure in a paper can carry a URL. Someone who opens it — a referee, a
replicator, you in eighteen months — sees the result, the code that produced
it, and the code and environment of every step behind it. No account, no
install, no notebook.

```bash
strata artifact publish nb_paper_cell_c2_var___display__0 --title "Figure 3"
```

```
nb_paper_cell_c2_var___display__0@v=1 is public at /p/ocxQj-toxGttZYyRl-Zf9...

Anyone with that link can read the artifact, its source, and the
source and environment of every step behind it. That is the point,
and it is worth knowing before sending the link:
  - nb_paper_cell_c2_var___display__0@v=1
  - nb_paper_cell_c1_var_rows@v=1

Withdraw it with: strata artifact unpublish ocxQj-toxGttZYyRl-Zf9...
```

Every cell output is already an artifact, and a plot is no exception — it is
stored under its own id with a provenance hash binding it to its inputs, its
source, and the environment it ran in. Publishing does not create any of that.
It grants read access to one version of it.

## What the page claims, and what it does not

This matters more here than in most features, because the reader is being
asked to trust what they see.

**Transparency.** The page shows the code, the inputs, the environment and the
author as recorded *when the bytes were produced* — not as they look now. A
cell edited after the run does not change what its artifact reports.

**Integrity.** A digest of the bytes is recorded at publication, and
`/p/<token>/verify` re-reads them and compares. That proves the bytes have not
changed since they were published. It does not prove they were honestly
produced in the first place — nothing a server can compute about its own
storage could establish that.

**Reproduction is not claimed.** Re-running the computation and getting the
same result is the strongest thing anyone could want, and it is not something
Strata asserts. Random seeds, thread counts, floating-point accumulation order
and unavailable input data each break it on their own. So there is no green
check anywhere on the page: in a research context a checkmark is read as
"someone reproduced this", and a badge that can be wrong is worse than no
badge. The page says what it checked, in words. Re-running is left to the
reader — the source and environment shown are what it takes.

## Publishing exposes the chain

This is the feature working as intended, and it is the thing to think about
before sending a link. The page shows the source and environment of every
upstream step, because a plot whose ancestry is hidden answers nothing.

Upstream cell source can name private dataset paths, internal table names, or
credential *names*. So publishing is explicit, one artifact version at a time,
never a switch on a whole notebook — and the CLI prints the full list of steps
the link will expose before you use it.

Upstream **data** is never served. The page describes the steps; only the
published artifact's own bytes are downloadable, at `/p/<token>/data`.

## Withdrawing

```bash
strata artifact unpublish <token>
```

The link then reports that it was withdrawn, rather than 404ing — a reader
chasing a footnote deserves that answer rather than one that reads like a typo.
The token is never reissued for other content, so a URL already in print fails
closed instead of quietly starting to resolve to something else.

## Reachability

The URL is on your Strata server, so it resolves for as long as that server is
reachable to the reader. For a lab or team server that is usually what you
want. A link printed in a published paper has a longer life than most servers,
and an archival export — a self-contained bundle suitable for Zenodo or OSF —
is the honest answer to that; it is not built yet.

## HTTP

| Route | Auth | Purpose |
| --- | --- | --- |
| `POST /v1/artifacts/{id}/v/{n}/publish` | yes (`artifacts:publish`) | Mint a link. Idempotent — republishing returns the existing token. |
| `DELETE /v1/publications/{token}` | yes (`artifacts:publish`) | Withdraw. |
| `GET /v1/publications` | yes | List this tenant's live links. |
| `GET /p/{token}` | **no** | The page. |
| `GET /p/{token}/data` | **no** | The published bytes. |
| `GET /p/{token}/verify` | **no** | Re-read and compare against the recorded digest. |
| `GET /v1/publications/{token}` | **no** | The same record as JSON. |

The unauthenticated routes are exempt from the auth and tenant middleware by
path, and only for `GET`. The token is the credential: it exists only because
someone with authority over that artifact minted it for one version. Minting,
withdrawing and listing all stay behind the gate.
