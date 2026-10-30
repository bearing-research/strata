# @name train_model
# @timeout 900
"""Train a tip-amount MLP on the materialized features, one checkpoint per epoch.

Each epoch saves the model, the optimizer, the shuffle generator's state and the
metrics through vestigo.torch's HostedRun: torch_checkpointing writes the files and
the server publishes them as the epoch's checkpoint, which vestigo.torch's
`Ledger.load` reads outside Strata too. A run that dies (crash, cancel, server
restart) continues from its last epoch on the next run and ends with the weights an
uninterrupted run would have. The best epoch so far is the `model` head.
"""

import hashlib

import numpy as np
import torch
from torch import nn
from vestigo.torch import HostedRun

FEATURES = [
    "pickup_hour",
    "pickup_dow",
    "duration_min",
    "passenger_count",
    "trip_distance",
    "pu_location_id",
    "do_location_id",
    "fare_amount",
]
TARGET = "tip_amount"
EPOCHS, BATCH = 12, 512

params = {"hidden": 64, "lr": 1e-3, "weight_decay": 1e-5, "seed": 42}

rows = np.random.default_rng(params["seed"]).permutation(len(features_df))
n_test = len(rows) // 5
test_rows, train_rows = rows[:n_test], rows[n_test:]
X = features_df[FEATURES].to_numpy(np.float32)
y = features_df[TARGET].to_numpy(np.float32)
feature_mean = X[train_rows].mean(axis=0)
feature_std = X[train_rows].std(axis=0) + 1e-6
X_test, y_test = X[test_rows], y[test_rows]
x_train = torch.from_numpy((X[train_rows] - feature_mean) / feature_std)
t_train = torch.from_numpy(y[train_rows])
x_test = torch.from_numpy((X_test - feature_mean) / feature_std)

torch.manual_seed(params["seed"])
model = nn.Sequential(
    nn.Linear(len(FEATURES), params["hidden"]),
    nn.ReLU(),
    nn.Linear(params["hidden"], params["hidden"]),
    nn.ReLU(),
    nn.Linear(params["hidden"], 1),
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
)
shuffle = torch.Generator().manual_seed(params["seed"])

run = HostedRun(strata)
state = run.restore(
    {
        "model": model,
        "optimizer": optimizer,
        "shuffle": shuffle.get_state(),
        "metrics": {"epoch": 0, "r2": None, "best_r2": None},
    }
)
shuffle.set_state(state["shuffle"])
metrics = state["metrics"]
if metrics["epoch"]:
    print(f"resumed after epoch {metrics['epoch']}")


def holdout_r2() -> float:
    with torch.no_grad():
        pred = model(x_test).squeeze(1).numpy()
    return float(1 - ((pred - y_test) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())


loss_fn = nn.MSELoss()
for epoch in range(metrics["epoch"] + 1, EPOCHS + 1):
    model.train()
    for batch in torch.randperm(len(x_train), generator=shuffle).split(BATCH):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train[batch]).squeeze(1), t_train[batch])
        loss.backward()
        optimizer.step()
    model.eval()
    r2 = holdout_r2()
    best = metrics["best_r2"] is None or r2 > metrics["best_r2"]
    metrics = {"epoch": epoch, "r2": r2, "best_r2": r2 if best else metrics["best_r2"]}
    sequence = run.save(
        epoch,
        {"model": model, "optimizer": optimizer, "shuffle": shuffle.get_state(), "metrics": metrics},
    )
    if best:
        strata.head("model", sequence)
    print(f"epoch {epoch}: holdout r2 {r2:.4f}")

run.close()

# Equal across an uninterrupted run and one resumed from any checkpoint.
weights = b"".join(p.detach().numpy().tobytes() for p in model.parameters())
print(f"weights sha256 {hashlib.sha256(weights).hexdigest()[:16]}")
print(f"trained on {len(x_train):,} rows, holdout {len(x_test):,} rows")
print(f"params: {params}")
