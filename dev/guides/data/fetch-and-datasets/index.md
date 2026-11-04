# Fetch and datasets

## @fetch

Declare a URL the cell reads. The bytes are downloaded into the notebook's
cache and **their digest becomes part of the cell's provenance**, so the cell
and its downstream cells go stale when the URL starts
serving something else - instead of the URL
string being in the source hash and the bytes being in nothing.

```python
# @fetch zones https://example.org/taxi_zones.csv
import pandas as pd

lookup = pd.read_csv(zones)
```

`<name>` is injected as a `pathlib.Path` to the downloaded file, which keeps the
URL's file name, so code that looks at the extension still works. Like a mount
variable it lives only in the declaring cell. In an R cell it is the path as a
string. A `@loop` cell cannot declare a fetch, because each iteration is a
separate run; fetch in an upstream cell instead.

Format: `# @fetch <name> <url> [sha256=<digest>] [refetch=never|stale|always]`.

- **`sha256=<digest>` pins the bytes.** The digest is the fingerprint, checked
  without the network while the cached copy matches. If the URL serves anything
  else, the cell fails with both digests and the error code
  `fetch_pin_mismatch` (`error_code` on the WebSocket `cell_error` frame, the
  REST and MCP run results, and the `strata run --format json` entry), so a
  client can tell a moved pin from any other failure without reading the text.
  `strata cell pin-fetch <notebook_dir> <cell_id> [NAME ...]` writes the pin for
  you: it sets `sha256=` on each named fetch (all of the cell's by default) to
  the digest of the bytes the notebook last downloaded, keeping the line's other
  options. Run the cell first so there are bytes to pin. After a mismatch,
  running it again accepts what the URL serves now.
- **`refetch=stale`** (default) checks the URL with a conditional GET
  (`ETag` / `Last-Modified`): at most once a minute while staleness is being
  recomputed, and always right before the cell runs, so what a run records is
  what the URL served then.
- **`refetch=never`** uses the cached bytes once there are any.
- **`refetch=always`** downloads again on every check, ignoring validators.

The first download happens only when the cell runs: opening or editing a
notebook sends no request to a URL it has never read, and the cell is never
shown as ready before that first run.

Each artifact the cell stores records the URL and the digest of the bytes the
run read as one of its inputs, and when those bytes were downloaded. That
record is how a [publication](https://bearingresearch.com/strata/dev/notebook/publishing/) page lists the URL under
"External inputs" with its digest and retrieval time, and its RO-Crate gives
the URL's `File` node an `sdDatePublished`. A snapshot export lists every fetch with whether it is
pinned, so an unpinned one can be flagged before the snapshot is shared, and
with `--include all` carries the fetched bytes, so a pinned or `refetch=never`
fetch runs on the importing machine without its URL.

Only `http` and `https` are fetched. In service mode every redirect hop passes
the same guard as a worker's URLs: no private, loopback or link-local address
unless the host is listed in `STRATA_NOTEBOOK_FETCH_ALLOWED_HOSTS`. The
connection goes only to an address that passed the check, so a name cannot
pass with a public address and then connect to `127.0.0.1`.

When `HTTPS_PROXY` or `HTTP_PROXY` covers a URL, a service-mode fetch goes
through that proxy instead. Each hop still passes the address check first, and
the proxy is sent the host name, so an egress proxy that admits hosts by name
decides. It also resolves the name itself, so it should refuse non-public
addresses too. Hosts in `NO_PROXY` connect directly, held to the checked
address.

Personal mode allows private addresses, so `http://localhost:8000/data.csv`
works on your own machine, and uses the proxy settings as usual. A URL that
cannot be checked leaves the cell idle, since nothing says the result is wrong, and running
it fails with the reason. A cell with `@fetch` runs on its own in Run All rather
than in a batch.

On a remote worker the server does the fetching: the bytes travel with the
cell's other inputs, uploaded on the direct transport and staged behind a
signed URL on the signed one, so the worker never contacts the URL and needs no
route to it. The worker must be running this version of Strata or later. An
older one does not recognise the input and the cell fails with a `NameError`
on `<name>`.

## @dataset

Declare a registry name the cell reads. The name resolves to one artifact
version, and **that version becomes part of the cell's provenance**, so when
`champion` moves, a cell that reads the team's champion model and its downstream
cells go stale. Resolving
the name with the `strata` client inside the cell records nothing: the name is
in the source hash and the version is in nothing.

```python
# @dataset model taxi/model@champion
predictions = model.predict(features)
```

Format: `# @dataset <name> <registry-name>[@<alias>|@v=<n>]`.

- **`taxi/model@champion`** follows the alias. The cell goes out of date when
  the alias points somewhere else.
- **`taxi/model`** follows the name pointer the same way.
- **`taxi/model@v=3`** pins version 3 of the artifact the name points at. A
  pinned dataset never makes the cell out of date, as `snapshot=` does for
  `@table`.

The server resolves the name in its own registry, or in the team store when
`STRATA_NOTEBOOK_REMOTE_STORE_URL` is set. The version is copied into the
notebook's own store, keeping its id and version, and `<name>` is bound to its value like an upstream variable: a
table as a DataFrame, a JSON value or a pickled object as itself. Artifacts
written outside a notebook (a core transform, `strata.put`) are Arrow tables.
A promoted [`# @output_dir`](https://bearingresearch.com/strata/dev/reference/annotations/python/#output_dir) folder arrives as the folder's `Path`,
its files copied with it. Anything else, such as an image, arrives as a
`pathlib.Path` to its bytes.

Each artifact the cell stores records the name and the version it resolved to
as an input, so lineage, in the dashboard and in `strata artifact lineage`,
walks from a downstream result through the cell to the named version. The copy
keeps that version's own inputs, and the steps behind it, up to ten deep, are
copied with it, so the walk goes on past the dataset to whatever made it. A
step the registry no longer holds ends the walk there, as it does in the
registry. The Registry tab lists, beside each name, the cells that read it.

The registry is asked at most once a minute while staleness is being
recomputed, and always right before the cell runs. A name that does not
resolve leaves the cell idle, and running it fails with the reason. A cell
with `@dataset` runs on its own in Run All rather than in a batch. `@loop` and
R cells cannot declare one; read the dataset in an upstream Python cell.
