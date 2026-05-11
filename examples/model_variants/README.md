# Model Variants — switch classifiers without forking the notebook

The same iris pipeline as `iris_classification`, but with three
alternative training cells grouped as variants of the same DAG slot.
Pick a classifier from the variant tabs and the rest of the notebook
re-runs against it; switch back and the previous result is a cache hit.

## What it shows

- **Variant cells.** `train_logreg`, `train_rf`, and `train_gbm` all
  carry `# @variant classifier <name>` and define the same contract
  (`model`). Only the active variant participates in the DAG.
- **Switching is cheap.** Each variant has its own provenance hash, so
  re-running an already-trained variant is a cache hit. Flip-flopping
  between two variants is free after each has run once.
- **Strict contract.** All variants must produce the same value
  bindings (imports don't count — they're scaffolding). If one variant
  accidentally adds an extra value, you get a
  `variant_contract_mismatch` diagnostic — the contract is what makes
  downstream cells correct under any selection.

## Cells

| Cell | What it does |
|---|---|
| `load_data` | Loads iris into `df` + `feature_names`. |
| `train_test` | 80/20 stratified split. |
| `train_logreg` | **Variant `classifier=logreg`** — `LogisticRegression`. Active by default. |
| `train_rf` | **Variant `classifier=rf`** — `RandomForestClassifier`. |
| `train_gbm` | **Variant `classifier=gbm`** — `GradientBoostingClassifier`. |
| `evaluate` | Test accuracy + per-class precision/recall against whichever `model` is active. |
| `confusion` | Confusion-matrix heatmap for the active classifier. |

## Running

From the project root:

```bash
uv run strata-server --host 127.0.0.1 --port 8765
```

Open `examples/model_variants` from the Strata home page. The
classifier group renders as a tab strip on the train cell — click a
tab to switch.

## Try this

- Run the notebook on `logreg`, then click the `rf` tab. `evaluate`
  and `confusion` go stale; re-run them to see the random-forest
  numbers. Click `logreg` again — both downstream cells become cache
  hits.
- Add a fourth variant: create `cells/train_svc.py` with
  `# @variant classifier svc` and `model = SVC(...)`. The tab appears
  immediately after the file is saved and the source is parsed.
- Edit a variant cell to define an extra variable (say,
  `feature_importance = ...`). The header pill flags
  `variant_contract_mismatch` — siblings disagree on what they
  expose. Remove the extra binding to clear it.

## Notes

The active selection is committed in `notebook.toml`'s
`[[variant_group]]` table — flipping variants from the UI shows up as a
git diff on that one line. That's intentional: the notebook records
which experiment you ran.
