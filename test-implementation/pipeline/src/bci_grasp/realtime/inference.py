"""Sliding-window online inference.

Scaffold only. Loads a trained model, buffers the latest ``tmax - tmin``
seconds of EEG from the LSL inlet, runs ``predict_proba``, and emits a
GRASP / REST command over serial via ``deployment.arduino_serial`` subject
to the decision threshold + dwell hysteresis in configs/deployment.yaml.

State machine (informal):

    BUFFER_FILLING → CLASSIFY → DECIDE → COMMAND? → BUFFER_FILLING

When the buffer has >= ``tmax - tmin`` seconds of EEG:
    p = model.predict_proba(window)[:, 1]   # P(MI)
    if p > threshold: candidate = GRASP
    elif p < 1 - threshold: candidate = REST
    else: candidate = hold previous
    if (candidate != current) and (dwell since last flip > min_dwell_s):
        send(candidate)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def run_realtime_loop(
    model_path: Path,
    lsl_stream_name: str,
    serial_port: str,
) -> None:
    """Main inference loop. Runs until interrupted (Ctrl-C) or stream dies.

    Parameters
    ----------
    model_path : Path
        Joblib-serialized model (``models/*.joblib``).
    lsl_stream_name : str
        LSL EEG stream to consume.
    serial_port : str
        Serial device to command the Arduino with (e.g. /dev/cu.usbmodem1101).
    """
    raise NotImplementedError(
        "Scaffold. Implement after offline training produces a working model."
    )
