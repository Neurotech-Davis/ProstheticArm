"""LSL marker outlet for the stimulus GUI.

Same pattern as ``reference/simple_bci/task.py::lsl_mrk_outlet`` — opens a
single-channel string-typed marker stream that LabRecorder picks up and
saves into the XDF alongside the EEG.

Labels pushed match configs/task.yaml:
    - "MI" / "Rest" at t=0 s of each trial (the "time mark stamps").
    - "run_start" / "run_end" around each run.
    - "die" to signal the backend (if any) to terminate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pylsl


def create_marker_outlet(
    name: str = "Task_Markers", stream_type: str = "Markers"
) -> "pylsl.StreamOutlet":
    """Open a 1-channel string LSL outlet.

    Why string-typed: human-readable markers are the path of least resistance
    for XDF post-processing. The dataset eventually maps them to integer
    codes (7 = MI, 9 = Rest) downstream.
    """
    raise NotImplementedError(
        "Implement: info = pylsl.StreamInfo(name, stream_type, 1, 0, pylsl.cf_string, 'ID_...'); "
        "return pylsl.StreamOutlet(info, 1, 1)."
    )
