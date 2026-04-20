"""Metrics: accuracy, ROC-AUC, confusion matrix.

All functions take ``y_true``, ``y_pred`` (+ optionally ``y_prob`` for AUC)
and return plain Python values / numpy arrays so they serialize cleanly to
the run's sidecar JSON.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified trials.

    Chance is 0.5 for the balanced MI/Rest task. Published single-trial
    accuracies on this dataset land in the 0.65–0.75 range — treat
    anything < 0.6 as likely broken.
    """
    raise NotImplementedError("Implement: sklearn.metrics.accuracy_score.")


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under ROC curve.

    Requires the classifier's probability / decision-function output for the
    positive class (MI = 1). More informative than accuracy when the chosen
    threshold isn't 0.5 (see configs/deployment.yaml decision_threshold).
    """
    raise NotImplementedError("Implement: sklearn.metrics.roc_auc_score.")


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix: rows = true class, cols = predicted.

    Convention: index 0 = Rest, index 1 = MI.
    """
    raise NotImplementedError(
        "Implement: sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])."
    )
