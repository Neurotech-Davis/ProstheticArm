"""Cross-validation splitters.

Primary: Leave-One-Subject-Out (LOSO). 10 subjects → 10 folds. This is the
honest test — we care about whether the model generalizes to a new user
without per-user calibration.

Secondary: stratified k-fold pooled across all subjects. Same subject appears
in train and test, so accuracy is usually 10–20 points higher than LOSO. Keep
it as a sanity-check upper bound, not as the headline metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator


def loso_splits(subject_ids: np.ndarray):
    """Yield (train_idx, test_idx) pairs for Leave-One-Subject-Out CV.

    Parameters
    ----------
    subject_ids : ndarray of shape (n_trials,)
        Subject label (any hashable) for each trial. Use the ``groups``
        argument of ``sklearn.model_selection.LeaveOneGroupOut``.

    Yields
    ------
    (train_idx, test_idx) : (ndarray, ndarray)
    """
    raise NotImplementedError(
        "Implement with sklearn.model_selection.LeaveOneGroupOut."
    )


def kfold_splits(n_splits: int = 5, shuffle: bool = True, random_state: int | None = 42):
    """Return a configured StratifiedKFold splitter.

    Stratified so MI/Rest proportions stay balanced in each fold even if trial
    rejection drops one class more than the other.
    """
    raise NotImplementedError(
        "Implement: return sklearn.model_selection.StratifiedKFold(...)."
    )
