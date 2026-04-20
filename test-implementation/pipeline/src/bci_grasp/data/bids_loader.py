"""BIDS loader for OpenNeuro ds003810 (MI vs Rest).

Thin wrapper around ``mne_bids.read_raw_bids`` that:

1. Builds a ``BIDSPath`` for a given subject + run.
2. Returns an ``mne.io.Raw`` object with events attached via annotations.
3. Maps the dataset's GDF-style event codes (7 = MI, 9 = Rest) to the class
   labels used by the classifier (1 = MI, 0 = Rest).

Why a wrapper: ``mne_bids`` is already good, but every script would otherwise
re-implement the same config→path plumbing and event-code mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mne


TASK_NAME = "MIvsRest"


def build_bids_path(bids_root: Path, subject: str, run: int):
    """Construct the BIDSPath for a single subject/run EEG recording.

    Parameters
    ----------
    bids_root : Path
        Absolute path to the dataset root (the dir containing
        ``dataset_description.json``).
    subject : str
        Subject label without the ``sub-`` prefix, e.g. ``"02"``.
    run : int
        Run index (0 = physical movement anchor, 1-4 = MI).

    Returns
    -------
    mne_bids.BIDSPath
    """
    raise NotImplementedError(
        "Implement using mne_bids.BIDSPath(subject=subject, task=TASK_NAME, "
        "run=run, datatype='eeg', root=bids_root, suffix='eeg', extension='.edf')."
    )


def load_run(bids_root: Path, subject: str, run: int) -> "mne.io.Raw":
    """Load a single subject/run as an MNE Raw with annotations.

    Event mapping (from ds003810 task-MIvsRest_events.json):
        raw event value 7  →  annotation description "MI"   (class 1)
        raw event value 9  →  annotation description "Rest" (class 0)

    The raw EDF stores these as ``OVTK_GDF_Right`` / ``OVTK_GDF_Tongue`` — the
    OpenViBE/GDF labels are misleading. Trust the code-to-label map above.

    Parameters
    ----------
    bids_root, subject, run :
        See ``build_bids_path``.

    Returns
    -------
    mne.io.Raw
        Preloaded Raw object. Caller is responsible for filtering, picking,
        and epoching.
    """
    raise NotImplementedError(
        "Implement: build BIDSPath → mne_bids.read_raw_bids(bids_path) → "
        "rename annotations '7'→'MI', '9'→'Rest' if needed → return raw.load_data()."
    )


def load_subject(bids_root: Path, subject: str, runs: list[int]) -> list["mne.io.Raw"]:
    """Load multiple runs for one subject. Order matches ``runs``.

    Returns a list rather than concatenating, so callers can apply artifact
    rejection per-run (artifact stats vary across runs).
    """
    return [load_run(bids_root, subject, r) for r in runs]
