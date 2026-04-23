"""Run the PsychoPy stimulus GUI (Graz MI-vs-Rest paradigm).

Launches a PsychoPy window, opens an LSL marker outlet (``Task_Markers``),
and runs one of:
  - run 0: physical movement anchor (kinesthetic reference).
  - runs 1-4: pure motor imagery.

Paired with LabRecorder + the EEG LSL stream for offline data collection.

Usage:
    conda activate psychopy_env
    python scripts/run_task.py --subject 01 --run 1
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="Subject label (e.g. '01')")
    parser.add_argument("--run", type=int, required=True, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    _ = args

    raise NotImplementedError(
        "Stub. Implement: load configs/task.yaml → create_marker_outlet → open "
        "psychopy.visual.Window → run_paradigm(run_index, outlet, window, cfg) → cleanup."
    )


if __name__ == "__main__":
    main()
