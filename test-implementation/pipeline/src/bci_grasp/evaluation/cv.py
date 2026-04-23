"""Cross-validation splitters.

Primary: Leave-One-Subject-Out (LOSO). With N subjects → N folds, each fold
trains on N-1 subjects and tests on the held-out one. This is the honest
test — we care about whether the model generalizes to a new user without
per-user calibration.

Secondary: stratified k-fold pooled across all subjects. Same subject appears
in train and test, so accuracy is usually 10-20 points higher than LOSO. Keep
it as a sanity-check upper bound, not as the headline metric.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold


def loso_splitter() -> LeaveOneGroupOut:
    """Return a ``LeaveOneGroupOut`` splitter.

    Pair with a ``groups=subject_ids`` ndarray of shape ``(n_trials,)`` when
    calling ``cross_validate`` / ``split``.
    """
    return LeaveOneGroupOut()


def kfold_splitter(
    n_splits: int = 5, shuffle: bool = True, random_state: int | None = 42
) -> StratifiedKFold:
    """Return a configured ``StratifiedKFold`` splitter.

    Stratified so MI/Rest proportions stay balanced in each fold even if trial
    rejection drops one class more than the other.
    """
    return StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )


def stack_subject_data(
    loaded: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack (subject_id, X, y) tuples into concatenated arrays for CV.

    Parameters
    ----------
    loaded : list of (subject_id, X, y)
        X has shape (n_epochs_i, n_channels, n_times); y has shape (n_epochs_i,).

    Returns
    -------
    (X, y, groups) : (ndarray, ndarray, ndarray)
        Shapes: (sum n_epochs_i, C, T), (sum n_epochs_i,), (sum n_epochs_i,).
        ``groups`` is the subject_id string repeated per trial — suitable
        for passing to ``LeaveOneGroupOut.split(X, y, groups)``.
    """
    Xs, ys, gs = [], [], []
    for sub_id, X, y in loaded:
        Xs.append(X)
        ys.append(y)
        gs.append(np.array([sub_id] * len(y)))
    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(gs, axis=0),
    )
