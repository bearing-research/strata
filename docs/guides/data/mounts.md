# Mounts

Attach a filesystem mount to the cell. Mounts provide read or read-write access to external storage (local paths, S3, GCS, Azure) during execution.

```python
# @mount raw_data s3://my-bucket/dataset ro
# @mount scratch file:///tmp/work rw
df = pd.read_parquet(raw_data / "events.parquet")
scratch / "summary.txt"  # → Path("/tmp/work/summary.txt")
```

**The mount name becomes a `pathlib.Path` variable in the cell's namespace.** No
`/mnt/<name>` directory convention - the variable directly references the
resolved local path (a local copy for remote URIs, the URI's local
filesystem path for `file://` URIs). Use standard `Path` operations: `/` for
joining, `.read_text()`, `.iterdir()`, etc. The mount wins over an earlier
cell that defines the same name, so the cell does not depend on that cell. The
same holds for the names `@table`, `@fetch` and `@dataset` bind.

**A read-only remote mount is copied once per machine.** Each remote file is
kept once, by URI and version (etag, size and modification time), in the
machine's mount cache, and the cell's path is a view of the prefix made of hard
links to those copies. When the prefix changes, only the new or changed files
are downloaded, and notebooks reading the same prefix share one copy when they
reach the store with the same storage options (a different credential value
makes its own copy, so a credential that can list a prefix but not read it never
gets another notebook's bytes). Its files and folders are read-only, since
every notebook using the view shares them. A run's result is keyed on the
listing its view was built from, so a prefix that changes while the run starts
is keyed on what the run read. The cache stays under a size bound
and drops views unused for some days, least recently used first; see
[`STRATA_NOTEBOOK_MOUNT_CACHE_DIR`](../../reference/configuration.md#notebook) and
the two settings after it. A worker keeps its cache across jobs. An `rw` mount
is still staged per notebook.

**A read-only local mount is keyed by what its files hold.** A cell's result is keyed
on the sha256 of each file under the mount, with its path relative to the mount, so a
rewrite that keeps the size and the modification time (`rsync -t`, `cp -p`, an
archive extracted again) makes the cell stale, and a `touch`, a `chmod` or a
`git checkout` of an unchanged file runs nothing. A file is read once: its digest is
kept in `.strata/hashes.sqlite` while the file's stat holds, so an unchanged mount
costs a `stat` per file. A file written less than 2 s before it was read is read again
next time. The staleness check reads at most 64 MiB of files it holds no digest for.
Past that the server hashes the mount in the background, and the cell shows stale
("Checking mount `raw`", with how much is hashed) until it is done; a cell whose files did not change is then
ready again, and a run in the meantime waits for that hash rather than starting its own.
Links to files are followed,
links to folders are not, and a file that cannot be read is keyed as unreadable. A
file that changed while it was read gets a key no later run matches, so that run's
result is never reused.

**A file written while the cell runs keeps that run to itself.** A local mount is not
copied: the cell reads the live folder, after its key was taken. So once a run
succeeds, its read-only local mounts are hashed again (mostly a `stat` per file, since
their digests were just kept; a file written less than 2 s before the run read it is
read again), and an `@each` instance's item file too; a loop cell's, once after its
last iteration. If one no longer holds what the key was taken from, the outputs and
displays are kept as a `@stochastic` run's are: under the run, where no later key finds
them, so a cell reading them keys on that run. The cell stays stale, says the mount
changed, and its console says which file moved
("`raw/part-3.csv` changed while the cell ran"). Putting the file back does not bring
those outputs back as a cache hit; the next run computes them again. Only the content is
compared, so a file written and put back before the run ends is not seen.

**A cell says how much it reads.** A badge shows what its read-only remote mounts,
`@table` inputs and `@fetch` URLs weighed when it last started ("reads 3.0 GB"), and
while it runs, how much of that moves onto the machine ("2.0 GB to read"), with each
input and what the machine already holds on hover. On a remote worker that share
is the worker's own cache, which the worker reports. A table counts its snapshot's
data files, and a pool cannot say which machine will run the cell, so those make a
total an upper bound ("up to 2.0 GB to read"). A size that is not known, such as a
pinned mount's, is not guessed. After a successful run, `.strata/reads/index.json` records
what each input moved and on which machine; see
[What a run moved](../../notebook/lake-aware-cells.md#what-a-run-moved).

![A cell that scans the table lake:taxi.rides, with a badge in its header reading "reads up to 27.5 MB" beside its cache badge.](../../assets/input-weight-light.png#only-light)
![A cell that scans the table lake:taxi.rides, with a badge in its header reading "reads up to 27.5 MB" beside its cache badge.](../../assets/input-weight-dark.png#only-dark)

Strata copies a remote mount in its own process, not the notebook's venv, so
its filesystem library goes where Strata is installed: the `s3` extra for
`s3://` (`s3fs`), `gcs` for `gs://` (`gcsfs`) and `azure` for `az://`
(`adlfs`), for example `uv tool install "strata-notebook[s3]"`.

**What a cell writes through an `rw` mount is recorded.** After a successful run,
the files it created or changed (path, size and sha256) are stored as the cell's
`__mount__<name>` record, linked to the inputs the run read, so its lineage shows
what produced them. On a remote worker the worker reports them. Each run adds a
version, since an `rw` cell never serves a cached result. Deleted files are not
recorded, and are not deleted from a remote mount either.

`.strata/writes/index.json` lists the same for each cell's last successful run, so
a host can list the files cells wrote to a source without reading each record:

```json
{
  "c1a2b3c4": {
    "ran_at": 1760889600000,
    "principal": null,
    "mounts": [
      {"mount": "out", "uri": "s3://bucket/out",
       "files": [{"path": "part-0.parquet", "size": 1048576, "sha256": "9f86d0..."}]}
    ]
  }
}
```

`principal` is the member the run was for (`null` in personal mode). A cell's entry
is replaced once its writes are synced back; a failed run, a failed sync-back
included, leaves it as it was, and a successful run that writes nothing (a cache
hit or a run-all batch too), or the cell's deletion, drops it. A `# @per_variant`
cell's entry lists every variant's files, merged per mount, once every variant has
succeeded; a path two variants wrote keeps the later one's. Loop, SQL and prompt
cells do not record writes, so a successful run of one drops the entry too.

Format: `# @mount <name> <uri> [ro|rw] [credential=<name>]`. Defaults to `ro` (read-only) if the mode is omitted. `credential=` names a [server-defined credential](../../reference/notebook-toml.md#named-credentials) instead of putting a secret in the cell. On a server that sets `STRATA_NOTEBOOK_CREDENTIAL_WRITERS`, an `rw` mount of `s3://`, `gs://` or `az://` must write through a credential its run's principal is listed for (its `credential=`, or the server's default for that scheme), or the cell fails before it runs; a credential that `STRATA_NOTEBOOK_CREDENTIAL_PATHS` restricts fails in any notebook outside its directories. The mount name must be a valid Python identifier (it's an injected variable).
