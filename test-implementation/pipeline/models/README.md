# models/

Trained model artifacts, one joblib file per training run plus a sidecar JSON.

Naming convention (set in `configs/model.yaml::artifact.run_id_format`):

    models/{YYYYMMDD_HHMMSS}_{model}_{eval}.joblib
    models/{YYYYMMDD_HHMMSS}_{model}_{eval}.json    # config + metrics + git SHA

Contents of the sidecar JSON:

```json
{
    "config": { ... snapshot of configs/*.yaml at train time ... },
    "metrics": { "accuracy_mean": 0.71, "roc_auc_mean": 0.78, "per_fold": [...] },
    "git_sha": "ccb08ee...",
    "timestamp_utc": "2026-..."
}
```

Files in this directory are gitignored (see pipeline/.gitignore); commit
only the sidecar JSON if you want to track historical results.
