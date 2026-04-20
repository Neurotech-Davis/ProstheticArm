"""Bandpower features (fallback / comparison to CSP).

Instead of learning spatial filters, just take the average band power in mu
and beta per channel. Simpler, no training data needed for the features, but
usually less accurate than CSP with few channels because it doesn't do any
cross-channel combination.

Useful for:
  - Sanity checks (if CSP can't beat bandpower, the pipeline is broken).
  - Explainability (per-channel ERD maps).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def bandpower(
    epochs_data: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    log: bool = True,
) -> np.ndarray:
    """Compute per-channel per-band power using Welch's method.

    Parameters
    ----------
    epochs_data : ndarray, shape (n_epochs, n_channels, n_times)
        Epoched EEG (already bandpass-filtered to a wider band).
    sfreq : float
        Sampling rate (125 for ds003810).
    bands : dict[str, (low, high)]
        e.g. ``{"mu": (8, 13), "beta": (13, 30)}``.
    log : bool
        Take natural log of power — approximately Gaussianizes the feature
        distribution, which helps LDA.

    Returns
    -------
    ndarray, shape (n_epochs, n_channels * n_bands)
        Flattened per-trial feature vector. Channel order matches the input;
        within each channel, bands appear in ``bands.keys()`` order.
    """
    raise NotImplementedError(
        "Implement with scipy.signal.welch: for each epoch, compute PSD per "
        "channel, integrate PSD between band edges, optionally log, then "
        "flatten (n_channels, n_bands) → (n_channels*n_bands,)."
    )
