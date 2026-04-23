"""Tests for CSP features + CSP→LDA pipeline.

Uses purely synthetic data so tests don't depend on the EEG dataset being
fetched. The synthetic task: class 1 has a 10 Hz burst with stronger
variance on channel 0 than channel 1; class 0 has the reverse. CSP should
pick this up effortlessly and LDA should achieve near-perfect accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from bci_grasp.features.csp import build_csp
from bci_grasp.models.lda_pipeline import build_lda_pipeline


def _synthetic_dataset(
    n_per_class: int = 50, n_channels: int = 6, n_times: int = 250, seed: int = 0
):
    """Two-class EEG-like dataset where variance structure differs between classes.

    For class 1: channel 0 is 3x the amplitude of channel 1.
    For class 0: channel 1 is 3x the amplitude of channel 0.
    CSP should find spatial filters that separate these two covariance structures.
    """
    rng = np.random.default_rng(seed)
    n_total = 2 * n_per_class

    X = rng.standard_normal((n_total, n_channels, n_times)) * 1e-6
    y = np.concatenate([np.ones(n_per_class, dtype=int), np.zeros(n_per_class, dtype=int)])

    # Class 1: big ch0, small ch1.
    X[:n_per_class, 0] *= 3.0
    # Class 0: big ch1, small ch0.
    X[n_per_class:, 1] *= 3.0
    return X, y


def test_csp_output_shape():
    csp = build_csp(n_components=4)
    X, y = _synthetic_dataset()
    Xt = csp.fit_transform(X, y)
    assert Xt.shape == (len(X), 4)


def test_csp_lda_pipeline_separates_synthetic_classes():
    """On an easy synthetic task, CSP+LDA should reach >0.9 with 5-fold CV."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X, y = _synthetic_dataset()
    pipe = build_lda_pipeline(csp_n_components=4, csp_reg="ledoit_wolf")
    scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    assert scores.mean() > 0.9, f"CSP+LDA failed on synthetic task: {scores}"


def test_lda_pipeline_predict_proba_shape():
    X, y = _synthetic_dataset(n_per_class=30)
    pipe = build_lda_pipeline(csp_n_components=4, csp_reg="ledoit_wolf")
    pipe.fit(X, y)
    probs = pipe.predict_proba(X[:5])
    assert probs.shape == (5, 2)
    # Each row should be a valid probability distribution.
    assert np.allclose(probs.sum(axis=1), 1.0)
