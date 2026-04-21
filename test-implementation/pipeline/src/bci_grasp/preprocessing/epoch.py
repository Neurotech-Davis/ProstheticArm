"""Cue-locked epoching.

An "epoch" is a fixed-length window of EEG aligned to a trial event. For this
dataset the event is the task-onset marker at t=0 s (pushed by the original
stimulus protocol when the red arrow / blank-Rest cue appears).

Default window [0.5, 2.5] s post-cue (see configs/preprocessing.yaml):
  - Skip the first 0.5 s to avoid the visual-evoked transient that briefly
    contaminates the sensorimotor band when the cue appears.
  - Stop at 2.5 s (task window is 4 s) to keep the window short enough
    for low-latency real-time inference later.

Label convention for the rest of the pipeline (used by models + metrics):
    Rest → 0, MI → 1.
"""

from __future__ import annotations

import mne


# Integer class labels used downstream. Keep in sync with features/metrics.
CLASS_REST = 0
CLASS_MI = 1


def epoch_cue_locked(
    raw: mne.io.BaseRaw,
    tmin: float = 0.5,
    tmax: float = 2.5,
    baseline: tuple[float, float] | None = None,
) -> mne.Epochs:
    """Slice ``raw`` into cue-locked MI and Rest trials.

    Assumes ``raw.annotations`` have already been normalized to "MI" and
    "Rest" labels (see ``data.bids_loader``).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Filtered Raw with "MI" / "Rest" annotations.
    tmin, tmax : float
        Epoch start/end in seconds relative to the event (t=0 = cue onset).
    baseline : (float, float) or None
        Baseline correction window. ``None`` disables baseline subtraction —
        the default for CSP-based pipelines (baseline correction on
        band-limited data is near no-op and can leak label info if the
        window overlaps cue-onset dynamics).

    Returns
    -------
    mne.Epochs
        Preloaded. ``epochs.events[:, -1]`` carries the integer class label
        (0 = Rest, 1 = MI). Use ``epochs["MI"]`` / ``epochs["Rest"]`` to
        select by class.
    """
    event_id = {"MI": CLASS_MI, "Rest": CLASS_REST}

    # Only these two annotation types become epochs — everything else
    # (trial boundaries, beeps, baselines) is ignored. ``regexp=None`` means
    # exact match against the event_id keys.
    events, found_id = mne.events_from_annotations(
        raw,
        event_id=event_id,
        regexp=None,
        verbose="ERROR",
    )
    # Defensive sanity check: make sure both classes were actually found.
    missing = set(event_id) - set(found_id)
    if missing:
        raise RuntimeError(
            f"Annotations missing labels {missing} — loader should normalize "
            "GDF strings before calling epoch_cue_locked."
        )

    return mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose="ERROR",
    )
