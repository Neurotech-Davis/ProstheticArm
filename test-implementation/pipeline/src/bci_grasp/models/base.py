"""Shared model I/O helpers.

Models are sklearn Pipelines; we serialize with joblib so the CSP transformer
(learned covariance matrices) is preserved alongside the classifier.

Each saved model gets a sidecar ``.json`` with:
  - the config snapshot used at train time
  - the reported metrics
  - the current git SHA

so a given ``models/<run_id>.joblib`` is fully self-documenting.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


def _git_sha() -> str:
    """Short SHA of HEAD, or 'unknown' if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_model(
    model: Any,
    path: Path,
    config: dict | None = None,
    metrics: dict | None = None,
) -> Path:
    """Persist a trained sklearn Pipeline to ``path`` (joblib) + sidecar JSON.

    Parameters
    ----------
    model : Any
        Fitted sklearn-compatible object.
    path : Path
        Target path ending in ``.joblib``.
    config : dict or None
        Snapshot of configs/*.yaml at train time (``load_config()``).
    metrics : dict or None
        Evaluation results (accuracy, AUC, per-fold).

    Returns
    -------
    Path
        The sidecar JSON path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "config": config or {},
                "metrics": metrics or {},
                "git_sha": _git_sha(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": path.name,
            },
            indent=2,
            default=str,
        )
    )
    return sidecar


def load_model(path: Path) -> Any:
    """Load a joblib-serialized model."""
    return joblib.load(Path(path))
