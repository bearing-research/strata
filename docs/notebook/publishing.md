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

## Where the artifact ends up

A notebook writes its cell outputs to its own `.strata/artifacts`; the server
serves whatever `artifact_dir` it was started with, `~/.strata/artifacts` by
default. Publishing therefore copies the artifact — and every step behind it,
since the page shows their code — into the store the server serves, and mints
the token there. The command says how many it moved:

```
Copied 4 artifacts into the server's store so the link resolves.
```

The copies keep their original ids, versions, provenance hashes, authors and
timestamps. An artifact that moves into a served store has to keep saying who
computed it and when; publishing is not a re-computation and must not read like
one.

`--here` skips the copy and publishes into whatever `--artifact-dir` names.
That link only resolves if the server serves that same directory — useful when
it does, misleading when it does not.

![The published page for a figure: the plot itself, then the artifact's id,
provenance hash and content digest, and below them the code that produced it
and every step behind
it.](../assets/publication-page-light.png#only-light)
![The published page for a figure: the plot itself, then the artifact's id,
provenance hash and content digest, and below them the code that produced it
and every step behind it.](../assets/publication-page-dark.png#only-dark)

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

## Archiving: the copy that needs no server

A hosted link resolves for as long as your server does. A URL printed in a
paper outlives most servers, so there is a second form:

```bash
strata artifact archive nb_paper_cell_c2_var___display__0 \
  --to ./figure3-bundle --title "Figure 3" --author "F. Li"
```

```
figure3-bundle/
├── index.html      the page — opens in a browser, no server, no external requests
├── artifact.png    the bytes
├── manifest.json   the same record, machine-readable
└── README.md       what it is and how to check it
```

Deposit the directory with Zenodo or OSF and cite the DOI. The archive's
retention promise then stands behind the link instead of yours.

The page is the same document as the hosted one, with one difference: it points
at the file beside it rather than at routes, and it tells the reader how to
check the bytes themselves:

```
sha256sum artifact.png
# 6803c74b80937b56ed0eb28f86995fbbe9559167ac6ce089525a1615afb0c6ca
```

Archiving is not publishing. It grants nobody access to a running server and
mints no link — it writes files you choose who to hand to.

## Crediting an author

The page carries two different facts, and they come from different places.

**Who published or archived it** is the byline under the title. `--author` sets
it on either command; in service mode a publish through the API uses the
authenticated principal instead. Omit it and there is simply no byline.

**Who computed it** is the *Computed by* row, and it comes from the artifact
itself — recorded when the cell ran, not when you published. A local run has no
authenticated identity to record, so that row usually reads "not recorded".
`--author` does not change it: crediting yourself for publishing a result is
not the same as the store attesting who produced it, and the page keeps them
apart deliberately.

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
