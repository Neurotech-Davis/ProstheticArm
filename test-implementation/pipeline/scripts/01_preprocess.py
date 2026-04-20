"""Stage 1: load BIDS → filter → epoch → save BIDS-Derivatives.

Reads ``configs/data.yaml`` + ``configs/preprocessing.yaml``, processes each
subject's runs 1-4 (MI runs; run 0 is the physical-movement anchor, not used
for the MI classifier), and writes:

    derivatives/bci-grasp-vs-rest/sub-XX/eeg/sub-XX_task-MIvsRest_epo.fif

Usage:
    conda activate psychopy_env
    python scripts/01_preprocess.py                 # all subjects
    python scripts/01_preprocess.py --subject 02    # one subject
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", help="Run a single subject (e.g. '02'). Default: all.")
    args = parser.parse_args()
    _ = args  # silence unused warning until implemented

    raise NotImplementedError(
        "Stub. Implement: load configs → for each subject: for each MI run: "
        "load_run → pick_deployment_channels → apply_bandpass → epoch_cue_locked → "
        "reject_by_amplitude → save under derivatives/bci-grasp-vs-rest/."
    )


if __name__ == "__main__":
    main()
