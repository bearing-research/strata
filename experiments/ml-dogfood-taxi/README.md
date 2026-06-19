# ml-dogfood-taxi

Phase 0 dogfood for the ML all-in-one investigation
(`docs/internal/design-ml-allinone.md`): run a real tabular ML project
end-to-end using **only Strata** — notebook + SDK + CLI as they exist today —
and log every gap in `friction-log.md`, in order hit, with severity.

## Scenario

Predict taxi tip amounts from NYC yellow-cab trips. Training data lives in a
local Iceberg warehouse; new data arriving = a new Iceberg snapshot.

- Month 1 (2024-01) loaded → snapshot S1; train + promote a model.
- Month 2 (2024-02) appended → snapshot S2; staleness detection → retrain →
  champion/challenger compare via lineage.

## Layout

```
data/                      # raw NYC TLC parquet (gitignored)
warehouse/                 # local Iceberg warehouse (gitignored)
setup_warehouse.py         # build warehouse, load month 1 (S1)
append_month2.py           # append month 2 (creates S2)
pipeline/                  # the Strata notebook (scan → features → train → eval → promote)
friction-log.md            # the deliverable: every gap, in order, with severity
```

## Rules of engagement

- Strata only: no DVC, no MLflow, no W&B, no manual parquet snapshots.
- Friction gets logged the moment it's felt, not reconstructed later.
- Workarounds are allowed (that's the point) but each one is a log entry.
