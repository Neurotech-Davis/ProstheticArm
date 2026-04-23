"""Trial-level artifact rejection.

The ds003810 paper used simultaneous EMG (Myoware) to reject trials
contaminated by actual muscle movement. This dataset copy does NOT include
EMG, so we fall back to EEG-only methods:

1. Amplitude thresholding (simple, fast, standard in BCI).
2. (Optional) autoreject — statistical per-channel thresholding.

Why trial rejection matters for BCI:
  - A handful of huge artifact trials will dominate CSP covariance estimation
    and wreck the spatial filters.
  - Dropping obvious bad trials usually improves LOSO accuracy more than any
    tuning of the classifier.
"""

from __future__ import annotations

import mne


def reject_by_amplitude(
    epochs: mne.Epochs, threshold_uv: float = 150.0
) -> mne.Epochs:
    """Drop epochs whose peak-to-peak amplitude on any channel exceeds ``threshold_uv``.

    Parameters
    ----------
    epochs : mne.Epochs
    threshold_uv : float
        Peak-to-peak threshold in microvolts. 100–200 uV is typical for
        wet-electrode EEG in the mu/beta band. Too low → drops real trials
        (the classifier sees mostly Rest); too high → no effect.

    Returns
    -------
    mne.Epochs
        A new Epochs object with bad trials dropped. Use ``epochs.drop_log``
        to inspect what was removed and why.
    """
    out = epochs.copy()
    # MNE's reject dict is in volts; convert from microvolts.
    out.drop_bad(reject={"eeg": threshold_uv * 1e-6}, verbose="ERROR")
    return out
