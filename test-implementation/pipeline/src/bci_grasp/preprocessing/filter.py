"""Task-band filtering.

The dataset was already hardware-filtered 0.5–45 Hz. This module applies the
task-specific bandpass (default 8–30 Hz, covering mu + beta) on top.

Why this band: motor imagery produces event-related desynchronization (ERD)
— a power decrease — primarily in mu (8–13 Hz) over contralateral sensorimotor
cortex, and secondarily in beta (13–30 Hz). A narrower band around mu only
(8–13) can work but usually hurts single-trial classification.

Why IIR (Butterworth) over FIR for BCI:
  - Lower latency: zero-phase FIR with the passband widths we need would
    require a very long kernel, which forces a long group delay and rules
    out tight real-time windows.
  - Flatter passband than a default MNE FIR for the same stop-band suppression.
"""

from __future__ import annotations

import mne


def apply_bandpass(
    raw: mne.io.BaseRaw,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    method: str = "iir",
    iir_params: dict | None = None,
) -> mne.io.BaseRaw:
    """In-place bandpass filter (Butterworth IIR by default).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preloaded Raw (``raw.load_data()`` must have been called — loader does this).
    l_freq, h_freq : float
        Band edges in Hz.
    method : {"iir", "fir"}
        "iir" preferred (see module docstring).
    iir_params : dict or None
        Passed through to MNE. Example: ``{"order": 4, "ftype": "butter"}``.
        None → MNE defaults to a 4th-order Butterworth.

    Returns
    -------
    mne.io.BaseRaw
        The same object, filtered in-place. Returned for chaining.
    """
    if iir_params is None:
        iir_params = {"order": 4, "ftype": "butter"}
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        iir_params=iir_params if method == "iir" else None,
        verbose="ERROR",
    )
    return raw


def apply_notch(raw: mne.io.BaseRaw, freqs: list[float]) -> mne.io.BaseRaw:
    """Notch-filter powerline (50 Hz for ds003810).

    Usually unnecessary once the task bandpass is 8–30 Hz (50 Hz is already
    well attenuated), but kept for exploratory plots of unfiltered data.
    """
    raw.notch_filter(freqs=freqs, verbose="ERROR")
    return raw
