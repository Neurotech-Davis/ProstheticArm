"""Channel selection.

Restricts the 15-channel ds003810 montage to the 6-channel subset used by the
planned OpenBCI Cyton deployment rig (C3, Cz, C4, F3, P3, Pz).

Why restrict at training time: we want the classifier's input distribution
to match deployment. Training on all 15 channels and then dropping 9 at
inference would break the learned spatial filters (especially CSP, whose
weights depend on the full channel covariance).
"""

from __future__ import annotations

import warnings

import mne


def pick_deployment_channels(
    raw: mne.io.BaseRaw, channels: list[str], strict: bool = True
) -> mne.io.BaseRaw:
    """Return a copy of ``raw`` restricted to ``channels`` (in that order).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object with the full channel set.
    channels : list[str]
        Channel names to keep, in the order the classifier expects.
    strict : bool
        If True, raise ``ValueError`` when any requested channel is missing
        from ``raw``. If False, silently drop missing ones and emit a warning.

    Returns
    -------
    mne.io.BaseRaw
        A copy with only ``channels`` present, in the requested order.
        Original raw is not mutated.

    Notes
    -----
    Uses ``raw.copy().pick(...)`` then ``reorder_channels(...)`` because
    ``pick`` with a list keeps MNE's internal order (by index in raw), not
    the caller's order. Downstream CSP requires a stable channel order
    across subjects/runs, so we force it here.
    """
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        if strict:
            raise ValueError(
                f"Requested channels missing from raw: {missing}. "
                f"Available: {raw.ch_names}"
            )
        warnings.warn(f"Skipping missing channels: {missing}", stacklevel=2)
        channels = [ch for ch in channels if ch in raw.ch_names]

    out = raw.copy().pick(channels)
    out.reorder_channels(channels)
    return out
