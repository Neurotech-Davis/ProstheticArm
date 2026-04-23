"""Stage 4 (scaffold only): real-time inference → Arduino.

Loads a trained model, opens an LSL inlet for live EEG, opens the Arduino
serial link, and enters the sliding-window inference loop defined in
``realtime.inference.run_realtime_loop``.

Usage:
    conda activate psychopy_env
    python scripts/04_run_realtime.py --model models/<run_id>.joblib \
        --stream OpenBCI_EEG --port /dev/cu.usbmodem1101
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to models/<run_id>.joblib")
    parser.add_argument("--stream", required=True, help="LSL EEG stream name")
    parser.add_argument("--port", required=True, help="Arduino serial port")
    args = parser.parse_args()
    _ = args

    raise NotImplementedError(
        "Scaffold. Wire together after offline training produces a working model."
    )


if __name__ == "__main__":
    main()
