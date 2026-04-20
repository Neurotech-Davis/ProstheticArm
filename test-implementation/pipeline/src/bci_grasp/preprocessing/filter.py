"""Task-band filtering.

The dataset was already hardware-filtered 0.5–45 Hz. This module applies the
task-specific bandpass (default 8–30 Hz, covering mu + beta) on top.

Why this band: motor imagery produces event-related desynchronization (ERD)
— a power decrease — primarily in mu (8–13 Hz) over contralateral sensorimotor
cortex, and secondarily in beta (13–30 Hz). A narrower band around mu only
(8–13) can work but usually hurts single-trial classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mne


def apply_bandpass(
    raw: "mne.io.Raw",
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    method: str = "iir",
    iir_params: dict | None = None,
) -> "mne.io.Raw":
    """In-place IIR bandpass filter (Butterworth by default).

    Parameters
    ----------
    raw : mne.io.Raw
        Preloaded Raw (``raw.load_data()`` must have been called).
    l_freq, h_freq : float
        Band edges in Hz.
    method : {"iir", "fir"}
        "iir" is preferred for BCI — lower latency, flatter passband.
    iir_params : dict or None
        Passed through to ``mne.filter.create_filter`` / ``raw.filter``.
        Example: ``{"order": 4, "ftype": "butter"}``.

    Returns
    -------
    mne.io.Raw
        The same object, filtered in-place. Return is a convenience for chaining.
    """
    raise NotImplementedError(
        "Implement: raw.filter(l_freq, h_freq, method=method, iir_params=iir_params)."
    )


def apply_notch(raw: "mne.io.Raw", freqs: list[float]) -> "mne.io.Raw":
    """Notch-filter powerline (50 Hz for ds003810).

    Usually unnecessary when the task bandpass is 8–30 Hz (50 Hz is already
    attenuated), but kept for exploratory plots of unfiltered data.
    """
    raise NotImplementedError("Implement: raw.notch_filter(freqs=freqs).")
