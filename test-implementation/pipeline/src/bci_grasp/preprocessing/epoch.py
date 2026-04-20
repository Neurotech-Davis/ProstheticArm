"""Cue-locked epoching.

An "epoch" is a fixed-length window of EEG aligned to a trial event. For this
dataset the event is the task-onset marker at t=0 s (pushed by the stimulus
GUI when the red arrow / blank-Rest cue appears).

Default window [0.5, 2.5] s post-cue (see configs/preprocessing.yaml):
  - skip first 0.5 s to avoid the visual-evoked transient that briefly
    contaminates the sensorimotor band
  - stop at 2.5 s (task window is 4 s) to keep the window short enough
    for low-latency real-time inference later
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mne


def epoch_cue_locked(
    raw: "mne.io.Raw",
    event_id: dict[str, int],
    tmin: float = 0.5,
    tmax: float = 2.5,
    baseline: tuple[float, float] | None = None,
) -> "mne.Epochs":
    """Slice ``raw`` into cue-locked trials.

    Parameters
    ----------
    raw : mne.io.Raw
        Filtered Raw with annotations or events.
    event_id : dict[str, int]
        Mapping label→code, e.g. ``{"MI": 7, "Rest": 9}`` from configs/data.yaml.
    tmin, tmax : float
        Epoch start/end in seconds relative to the event (t=0 = cue onset).
    baseline : (float, float) or None
        Baseline correction window. ``None`` disables baseline subtraction —
        the default for CSP-based pipelines.

    Returns
    -------
    mne.Epochs
        Trials × channels × samples. ``epochs.events[:, -1]`` carries the
        numeric class code; use ``epochs["MI"]`` / ``epochs["Rest"]`` to split.

    Notes
    -----
    The label→integer convention the classifier expects downstream:
        Rest → 0, MI → 1.
    Build ``y`` from ``epochs.events[:, -1]`` and the inverse of ``event_id``.
    """
    raise NotImplementedError(
        "Implement: mne.events_from_annotations(raw, event_id=event_id) → "
        "mne.Epochs(raw, events, event_id, tmin, tmax, baseline, preload=True)."
    )
