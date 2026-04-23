"""Metrics: accuracy, ROC-AUC, confusion matrix.

All functions take ``y_true``, ``y_pred`` (+ optionally ``y_prob`` for AUC)
and return plain Python values / numpy arrays so they serialize cleanly to
the run's sidecar JSON.

Class convention (see preprocessing.epoch):
    0 = Rest (negative class)
    1 = MI   (positive class)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified trials.

    Chance is 0.5 for the balanced MI/Rest task. Published single-trial
    accuracies on this dataset land in the 0.65-0.75 range — treat
    anything < 0.6 as likely broken.
    """
    return float(accuracy_score(y_true, y_pred))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under ROC curve.

    Requires the classifier's probability / decision-function output for the
    positive class (MI = 1). More informative than accuracy when the chosen
    threshold isn't 0.5 (see configs/deployment.yaml decision_threshold).

    If y_true has only one class (rare — degenerate LOSO fold), returns NaN.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix: rows = true class, cols = predicted.

    Index 0 = Rest, index 1 = MI.
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
