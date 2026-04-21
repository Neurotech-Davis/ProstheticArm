"""BIDS loader for OpenNeuro ds003810 (MI vs Rest).

The dataset is BIDS-EEG with EDF source files. There is **no** per-run
``*_events.tsv`` on disk, so event information must come from the EDF's own
annotation channel, which stores OpenViBE GDF stimulation labels as strings:

    raw annotation string   →   dataset label   →   class id
    "OVTK_GDF_Right"        →   "MI"            →   1
    "OVTK_GDF_Tongue"       →   "Rest"          →   0

The dataset's ``task-MIvsRest_events.json`` assigns codes 7 (MI) and 9 (Rest)
to those two GDF labels respectively. The integer codes are what the rest of
the pipeline (configs/data.yaml::event_id) uses, so this loader normalizes
the annotation descriptions to the human-readable labels ("MI" / "Rest")
before returning the Raw.

We intentionally do NOT go through ``mne_bids.read_raw_bids`` here: the
dataset's idiosyncratic event representation (GDF strings embedded in the
EDF, no events.tsv) means ``read_raw_bids`` would either skip events or
require us to materialize a transient events.tsv. Reading the EDF directly
+ renaming annotations is simpler and explicit.
"""

from __future__ import annotations

from pathlib import Path

import mne


TASK_NAME = "MIvsRest"

# Map raw EDF annotation strings to human-readable labels used downstream.
# Anything not in this dict is kept as-is (useful for debugging), but the
# pipeline only cares about MI and Rest.
GDF_TO_LABEL: dict[str, str] = {
    "OVTK_GDF_Right": "MI",
    "OVTK_GDF_Tongue": "Rest",
}


def edf_path(bids_root: Path, subject: str, run: int) -> Path:
    """Construct the absolute path to a subject/run EDF.

    Parameters
    ----------
    bids_root : Path
        Absolute path to the dataset root (the dir containing
        ``dataset_description.json``).
    subject : str
        Subject label without the ``sub-`` prefix, e.g. ``"02"``.
    run : int
        Run index (0 = physical movement anchor, 1-4 = MI).
    """
    return (
        Path(bids_root)
        / f"sub-{subject}"
        / "eeg"
        / f"sub-{subject}_task-{TASK_NAME}_run-{run}_eeg.edf"
    )


def load_run(bids_root: Path, subject: str, run: int) -> mne.io.BaseRaw:
    """Load a single subject/run EDF with annotations normalized.

    The returned Raw has:
      - All 15 EEG channels (see configs/data.yaml::all_channels).
      - Annotations where the GDF strings for MI and Rest trials have been
        renamed to "MI" and "Rest" respectively. All other annotation
        strings (trial boundaries, beeps, etc.) are preserved unchanged.
      - ``raw.load_data()`` already called so downstream filtering works.

    Parameters
    ----------
    bids_root, subject, run :
        See ``edf_path``.

    Returns
    -------
    mne.io.BaseRaw
    """
    path = edf_path(bids_root, subject, run)
    if not path.exists():
        raise FileNotFoundError(
            f"EDF not found: {path}. If this is a broken git-annex symlink, "
            "run `datalad get sub-{subject}` from the ds003810 root."
        )

    # verbose='ERROR' silences MNE's chatty per-load info; re-enable if
    # debugging a subject whose annotations look off.
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")

    # Unit fix-up.
    # The EDF's physical-dimension field is the non-standard string "Microv"
    # (see task-MIvsRest_channels.tsv), which MNE cannot map to a known SI
    # scale. It therefore keeps the raw physical values and labels them as
    # volts — so the data array is in microvolts but tagged V. Scale into
    # true volts so MNE's reject thresholds, PSD units, and topomap vmin/vmax
    # all behave normally downstream.
    raw.apply_function(lambda x: x * 1e-6, picks="eeg")

    # In-place rename: MNE's Annotations are stored in raw.annotations, and we
    # rewrite descriptions via set_annotations with a new Annotations object.
    # (There is no "rename" method; building a new object is the canonical way.)
    new_descriptions = [
        GDF_TO_LABEL.get(str(d), str(d)) for d in raw.annotations.description
    ]
    raw.set_annotations(
        mne.Annotations(
            onset=raw.annotations.onset,
            duration=raw.annotations.duration,
            description=new_descriptions,
            orig_time=raw.annotations.orig_time,
        )
    )
    return raw


def load_subject(
    bids_root: Path, subject: str, runs: list[int]
) -> list[mne.io.BaseRaw]:
    """Load multiple runs for one subject. Order matches ``runs``.

    Returns a list rather than concatenating, so callers can apply artifact
    rejection per-run (artifact stats vary across runs) and so we never
    accidentally merge RUN 0 (physical movement) with the MI runs.
    """
    return [load_run(bids_root, subject, r) for r in runs]
