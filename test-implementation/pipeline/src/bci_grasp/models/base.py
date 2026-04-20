"""Shared model I/O helpers.

Models are sklearn Pipelines; we serialize with joblib so the CSP transformer
(learned covariance matrices) is preserved alongside the classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_model(model: Any, path: Path) -> None:
    """Persist a trained sklearn Pipeline to ``path`` (joblib format).

    Writes a sidecar ``.json`` next to the model containing:
      - config snapshot (so the model can be reloaded without guessing bands)
      - metrics (accuracy / AUC from evaluation)
      - git SHA of the repo at save time

    Parameters
    ----------
    model : Any
        Fitted sklearn-compatible object.
    path : Path
        Target path ending in ``.joblib``.
    """
    raise NotImplementedError(
        "Implement with joblib.dump. Write sidecar JSON with config + metrics + SHA."
    )


def load_model(path: Path) -> Any:
    """Load a joblib-serialized model.

    Parameters
    ----------
    path : Path
        Source path ending in ``.joblib``.
    """
    raise NotImplementedError("Implement with joblib.load(path).")
