# Registry Dashboard

The registry dashboard is the **UI for promoting models and approving changes**
right inside the notebook — so promotion doesn't have to be code. It surfaces
the same names / aliases / tags / audit / lineage that the
[SDK and CLI](../core/registry.md) drive, backed by the identical audited routes.

!!! note "Personal mode only (today)"
    The registry routes are personal-mode only right now, so the dashboard
    **shows up automatically in personal mode** (the default `python -m strata`)
    and **hides itself in service mode**. If you don't see it, check your
    deployment mode.

This page is a step-by-step walkthrough. For the concepts (what a name, alias,
tag, or approval gate *is*), see [Artifacts & Model Registry](../core/registry.md).

## What you'll build

Publish a model from a cell, promote it to **champion**, watch the move land in
the registry with full audit + lineage — without leaving the notebook.

---

## 1. Open the notebook UI

Start the server and open it in a browser:

```bash
python -m strata          # serves the bundled UI on http://localhost:8765
```

Open **http://localhost:8765** and open (or create) a notebook.

## 2. Publish a model from a cell

The dashboard surfaces what your **cells publish** to the registry. Inside any
Python cell, use the [ambient `strata` client](cells.md#the-ambient-strata-client)
— it's already in the namespace, no import or setup:

```python
# ... you've trained `model` in this or an upstream cell ...
art = strata.put(model, name="taxi/tip-model")
```

Run the cell (Shift+Enter). `strata.materialize(..., name="taxi/tip-model")`
works the same way. The artifact lands in the registry and is **stamped with the
cell that produced it**.

## 3. The per-cell promote strip

A compact strip appears **right below the cell** that published:

```
⬡ taxi/tip-model  v1   [Promote ▾]   ⎘
```

- `⬡ taxi/tip-model v1` — the name and version your cell put.
- `[Promote ▾]` — promote this artifact (next step).
- `⎘` — open its lineage.

Promote where you trained the model, without scrolling anywhere.

## 4. Promote to champion

Click **`[Promote ▾]`** and choose **champion** (or **candidate**). A toast
confirms the result:

- **`✓ taxi/tip-model → champion`** — applied immediately (the normal case).
- **`⏳ champion change pending approval`** — the alias is *protected* (see
  step 6); the move is queued instead of applied.

`champion` / `candidate` are intent pointers — the post-stages model, not
`Staging`/`Production` enums. A name can hold both at once (champion *and* a
challenger candidate).

## 5. The Registry tab

Open the **bottom drawer** and click the **Registry** tab. It has three parts,
top to bottom:

1. **Pending-approval banner** — appears only when a protected-alias move is
   queued, with **Approve / Reject** buttons (the human gate, in the UI).
2. **Names table** — every registry name, each row showing its **alias chips**
   (`★champ`, `cand`), latest version, **tags**, a **`[Promote ▾]`** menu, and a
   **`⎘`** lineage button. This is the same data on the per-cell strip, but for
   *all* names — not just what the current notebook published.
3. **Audit timeline** (collapsible) — every name / alias / tag mutation, newest
   first, with who and from → to.

## 6. Approval gates (protected aliases)

To require a human for sensitive promotions, mark aliases protected when you
start the server:

```bash
STRATA_REGISTRY_PROTECTED_ALIASES=champion,production python -m strata
```

Now a promote to `champion` (step 4) returns **`⏳ pending`** instead of applying.
The **pending banner** appears at the top of the Registry tab; click **Approve**
to apply it (the approver becomes the audit actor) or **Reject** to discard it.
Unprotected aliases (`candidate`) still apply immediately.

## 7. View lineage

Click the **`⎘`** button on the strip or any names-table row. The **lineage
view** renders the provenance chain:

```
model ← features ← scan ← table @ snapshot
```

— the same chain `strata artifact lineage` prints on the CLI, as an interactive
view. It answers "which snapshot trained this model?" in one click.

---

## Troubleshooting

| Symptom | Why |
| --- | --- |
| **No Registry tab / no strip at all** | You're in **service mode**. The dashboard is personal-mode only today. |
| **Registry tab is empty** | Nothing's been published with a name yet. Run a cell with `strata.put(value, name="…")`. |
| **A cell ran but no per-cell strip** | The strip only shows artifacts a cell published via the ambient `strata` client (a *named* `put` / `materialize`). Artifacts created another way still appear in the **names table**, just not as a per-cell strip. |
| **Promote said "pending" unexpectedly** | That alias is in `STRATA_REGISTRY_PROTECTED_ALIASES` — approve it from the pending banner. |
| **Promote did nothing** | Setting an alias to the version it already points at is an idempotent **no-op** (the toast says `unchanged`). |

## See also

- [Artifacts & Model Registry](../core/registry.md) — the concepts and the
  SDK/CLI that drive the same registry.
- [The ambient `strata` client](cells.md#the-ambient-strata-client) — how cells
  publish.
- [Lake-Aware Cells](lake-aware-cells.md) — the `@table` flow that makes a
  training cell stale so you *re-promote* when new data lands.
