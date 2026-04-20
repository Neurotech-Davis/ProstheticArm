"""Channel selection.

Restricts the 15-channel ds003810 montage to the 6-channel subset used by the
planned OpenBCI Cyton deployment rig (C3, Cz, C4, F3, P3, Pz).

Why restrict at training time: we want the classifier's input distribution
to match deployment. Training on all 15 channels and then dropping 9 at
inference would break the learned spatial filters (especially CSP).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mne


def pick_deployment_channels(
    raw: "mne.io.Raw", channels: list[str], strict: bool = True
) -> "mne.io.Raw":
    """Return a copy of ``raw`` restricted to ``channels`` (in that order).

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with the full channel set.
    channels : list[str]
        Channel names to keep, in the order the classifier expects.
    strict : bool
        If True, raise when any requested channel is missing. If False,
        silently drop missing ones and warn.

    Returns
    -------
    mne.io.Raw
        A *copy* with only ``channels`` present, in the requested order.
        Original raw is not mutated.
    """
    raise NotImplementedError(
        "Implement: validate channels vs raw.ch_names → raw.copy().pick(channels) "
        "→ reorder via raw.reorder_channels(channels)."
    )
