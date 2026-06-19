# @name promote
"""Promote the trained model through the registry.

Uses the full registry surface: the model artifact is pushed with
lineage, named, tagged with its eval metrics, and proposed as the new
champion via an alias move — which the server may gate behind approval
when `champion` is protected (STRATA_REGISTRY_PROTECTED_ALIASES).
"""

import pickle

import pyarrow as pa


model_table = pa.table(
    {
        "model_pickle": pa.array([pickle.dumps(model)], type=pa.binary()),
        "framework": ["sklearn.HistGradientBoostingRegressor"],
    }
)

model_art = strata.put(
    inputs=[features_uri],
    transform={"executor": "train_hgbr@v1", "params": dict(params)},
    data=model_table,
    name="taxi/tip-model",
)

metrics_art = strata.put(
    inputs=[model_art.uri],
    transform={"executor": "evaluate@v1", "params": {"holdout": "20pct, seed 42"}},
    data=eval_metrics,
    name="taxi/tip-model-metrics",
)

# Tag the model version with its eval facts (queryable later)
mae = float(eval_metrics.loc[eval_metrics.metric == "mae", "value"].iloc[0])
r2 = float(eval_metrics.loc[eval_metrics.metric == "r2", "value"].iloc[0])
strata.set_tag(model_art.artifact_id, model_art.version, "mae", f"{mae:.4f}")
strata.set_tag(model_art.artifact_id, model_art.version, "r2", f"{r2:.4f}")
strata.set_tag(
    model_art.artifact_id, model_art.version, "trained_at_snapshot", str(scanned_snapshot)
)

# Propose this version as champion. If the alias is protected the server
# answers 202 pending and a human approves via the registry queue.
move = strata.set_alias(
    "taxi/tip-model", "champion", model_art.artifact_id, model_art.version
)
promotion_status = move.get("status", "applied")

promoted_uri = model_art.uri
print(f"model artifact:  {promoted_uri}")
print(f"metrics:         mae={mae:.4f} r2={r2:.4f}")
print(f"champion move:   {promotion_status}")
