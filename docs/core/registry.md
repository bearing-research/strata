# Artifacts & Model Registry

Every materialized result in Strata is an immutable, versioned **artifact**
with content-addressed provenance. The registry layer adds the pointers and
history that turn the artifact store into a lightweight model registry:
**names**, **aliases**, **tags**, an append-only **audit**, and optional
**approval gates** — all in the same SQLite metadata store, no extra
services.

## Names

A name is a mutable, human-readable pointer to one artifact version:

```python
client.materialize(inputs=[...], transform=..., name="taxi/tip-model")
client.set_name("taxi/tip-model", artifact_id, version)
resolved = client.resolve_name("taxi/tip-model")
```

Slash-namespaced names (`team/dataset/raw`) are the natural convention and
fully supported. A name tracks "the latest blessed build of this line";
every move is audited.

## Aliases

Aliases are intent pointers *on a name* — the post-stages registry model
(champion/candidate rather than Staging/Production enums):

```python
client.set_alias("taxi/tip-model", "champion", artifact_id, version)
client.set_alias("taxi/tip-model", "candidate", new_id, new_version)
client.resolve_alias("taxi/tip-model", "champion")
```

A name can hold any number of aliases. Aliases may pin a **superseded**
version (an old champion stays fetchable after a rebuild), and an aliased
artifact is protected from garbage collection. Setting an alias to the
version it already points at is a **no-op** — idempotent promote cells
re-run without spamming history.

In the artifact CLI, `name@alias` works anywhere a reference does:

```bash
strata artifact show taxi/tip-model@champion
strata artifact lineage taxi/tip-model@champion
strata artifact pull taxi/tip-model@champion --to model.arrow
```

## Tags

Key/value facts about one artifact version — recorded at promote time,
queryable later:

```python
client.set_tag(artifact_id, version, "mae", "1.226")
client.set_tag(artifact_id, version, "trained_at_snapshot", str(snapshot))
client.get_tags(artifact_id, version)
```

## Audit

Every name, alias, and tag mutation lands in an append-only audit table —
written **in the same transaction** as the mutation, so a change can never
land unrecorded. The audit answers "what did champion point to before?":

```bash
$ strata artifact audit taxi/tip-model
2026-06-07 06:01  alias_set         taxi/tip-model@champion  4861a747@v1 -> e3ea60b6@v1
2026-06-07 06:01  alias_approved    taxi/tip-model@champion
2026-06-07 06:01  alias_request_set taxi/tip-model@champion
```

Entries carry the actor (principal id when authenticated), the action, and
the from → to versions. `GET /v1/registry/audit` serves the same data; the
SDK exposes `get_registry_audit(name=..., artifact_id=...)`.

## Approval gates

Protect aliases whose moves should require a human:

```bash
STRATA_REGISTRY_PROTECTED_ALIASES=champion,production
```

Moves (and deletes) of protected aliases return **202 pending** instead of
applying. The queue is visible, and approval applies the move with the
approver as the audit actor — atomically with the pending-consumption, so
a crash can never swallow an approval:

```python
client.list_pending_changes()
client.approve_alias_change("taxi/tip-model", "champion")
client.reject_alias_change("taxi/tip-model", "champion")
```

```bash
strata artifact pending          # the review queue, from the CLI
```

If a pending change's target artifact disappears before approval, the
approve fails cleanly and the pending entry stays for an explicit reject.
Unprotected aliases apply immediately; the default is no gating.

## The promotion flow, end to end

The shape this is designed for — a notebook training pipeline whose last
cell promotes through the registry:

```python
# promote cell
model_art = client.put(inputs=[features_uri], transform=..., name="taxi/tip-model")
client.set_tag(model_art.artifact_id, model_art.version, "mae", f"{mae:.4f}")
move = client.set_alias("taxi/tip-model", "champion",
                        model_art.artifact_id, model_art.version)
print(move.get("status", "applied"))   # "applied" | "pending" | "unchanged"
```

New data lands in the lake → the [`@table` annotation](../notebook/annotations.md#table)
makes the training cells stale → the pipeline re-runs → promote proposes the
challenger → a human approves → and the whole history is reconstructible:

```bash
strata artifact lineage taxi/tip-model@champion
# model <- features <- scan <- table file://...#nyc.trips @ snapshot 2558063...
strata artifact audit taxi/tip-model
strata artifact verify    # store-wide blob/metadata consistency check
```

## Storage & durability

Registry state lives in the same `artifacts.sqlite` as artifact metadata
(WAL mode, transaction-per-mutation). Everything commits before the API
responds; the audit is in-transaction with its mutation; server restarts
are non-events. The CLI reads the store directly — server up or down.
