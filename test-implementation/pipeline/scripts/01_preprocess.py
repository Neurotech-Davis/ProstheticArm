"""Stage 1: load BIDS → filter → epoch → save BIDS-Derivatives.

Reads ``configs/data.yaml`` + ``configs/preprocessing.yaml``, processes each
subject's MI runs (runs 1-4; run 0 is the physical-movement anchor, not used
for the MI classifier), and writes one concatenated epochs file per subject:

    derivatives/bci-grasp-vs-rest/sub-XX/eeg/sub-XX_task-MIvsRest_epo.fif

Why concatenate runs: downstream the classifier uses the subject as the CV
group, so a single .fif per subject is what 02_train needs. Per-run metadata
is lost after concatenation — if we ever need to keep it, switch to writing
one .fif per run and concatenate at train time.

Usage
-----
    conda activate psychopy_env
    python scripts/01_preprocess.py                 # all subjects in data.yaml
    python scripts/01_preprocess.py --subject 02    # one subject
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne

from bci_grasp.artifacts.reject import reject_by_amplitude
from bci_grasp.config import load_config, resolve_repo_path
from bci_grasp.data.bids_loader import load_run
from bci_grasp.data.channel_select import pick_deployment_channels
from bci_grasp.preprocessing.epoch import epoch_cue_locked
from bci_grasp.preprocessing.filter import apply_bandpass

log = logging.getLogger("preprocess")


def derivatives_root() -> Path:
    """Return ``pipeline/derivatives/bci-grasp-vs-rest`` as absolute Path."""
    from bci_grasp.config import PIPELINE_ROOT

    return PIPELINE_ROOT / "derivatives" / "bci-grasp-vs-rest"


def preprocess_subject(subject: str, cfg_data: dict, cfg_pre: dict) -> Path:
    """Run the full preprocessing pipeline for a single subject.

    Loads every MI run, filters, epochs, rejects bad trials, concatenates
    across runs, writes the result to derivatives/.

    Returns
    -------
    Path
        The written ``*_epo.fif`` path.
    """
    bids_root = resolve_repo_path(cfg_data["bids_root"])
    channels = cfg_data["deployment_channels"]
    runs = cfg_data["runs_mi"]

    bp = cfg_pre["bandpass"]
    ep_cfg = cfg_pre["epoch"]

    all_epochs: list[mne.Epochs] = []
    for run in runs:
        raw = load_run(bids_root, subject, run)
        raw = pick_deployment_channels(raw, channels, strict=True)
        apply_bandpass(
            raw,
            l_freq=bp["l_freq"],
            h_freq=bp["h_freq"],
            method=bp["method"],
            iir_params=bp.get("iir_params"),
        )
        epochs = epoch_cue_locked(
            raw,
            tmin=ep_cfg["tmin"],
            tmax=ep_cfg["tmax"],
            baseline=ep_cfg.get("baseline"),
        )
        # Amplitude rejection: 150 uV matches dataset notes; tune if too
        # aggressive on a given subject.
        epochs = reject_by_amplitude(epochs, threshold_uv=150.0)
        n_mi = len(epochs["MI"])
        n_rest = len(epochs["Rest"])
        log.info("sub-%s run-%d kept %d MI + %d Rest", subject, run, n_mi, n_rest)
        all_epochs.append(epochs)

    combined = mne.concatenate_epochs(all_epochs, verbose="ERROR")
    out_dir = derivatives_root() / f"sub-{subject}" / "eeg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{subject}_task-MIvsRest_epo.fif"
    combined.save(out_path, overwrite=True, verbose="ERROR")
    log.info(
        "sub-%s → %s (%d epochs: %d MI, %d Rest)",
        subject,
        out_path.relative_to(out_path.parents[4]),  # relative to repo root
        len(combined),
        len(combined["MI"]),
        len(combined["Rest"]),
    )
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", help="Run a single subject (e.g. '02'). Default: all.")
    args = parser.parse_args()

    cfg_data = load_config("data")
    cfg_pre = load_config("preprocessing")

    subjects = [args.subject] if args.subject else cfg_data["subjects"]
    for sub in subjects:
        try:
            preprocess_subject(sub, cfg_data, cfg_pre)
        except FileNotFoundError as e:
            # sub-* not fetched via datalad yet — skip and keep going.
            log.warning("sub-%s skipped: %s", sub, e)


if __name__ == "__main__":
    main()
