"""Graz-based MI-vs-Rest paradigm.

Mirrors the structure of ``reference/simple_bci/task.py`` (ollie-d's LSL +
PsychoPy skeleton). Not implemented yet — stubs only. All timing and visual
parameters come from ``configs/task.yaml``.

Trial structure (t relative to task onset):
    t = -3 s   fixation cross appears
    t = -1 s   audible beep
    t =  0 s   visual stimulus + LSL marker pushed
                 MI   : red right-pointing arrow for 4 s
                 Rest : nothing (fixation stays on) for 4 s
    t = +4 s   stimulus removed
    then        inter-trial interval uniform in [2.5, 4.5] s

Per-run structure:
    20 MI + 20 Rest trials, shuffled balanced (not independent random draws).
    RUN 0 = physical movement (kinesthetic anchor).
    RUN 1-4 = pure motor imagery.

IMPORTANT: refresh_rate_hz in configs/task.yaml MUST match the monitor's
actual refresh rate. PsychoPy times events in frames, so a wrong refresh
rate makes every timing claim above a lie. Verify with
``win.getActualFrameRate()`` on first run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pylsl
    import psychopy.visual


def run_paradigm(
    run_index: int,
    marker_outlet: "pylsl.StreamOutlet",
    window: "psychopy.visual.Window",
    config: dict,
) -> None:
    """Execute one run of the paradigm.

    Parameters
    ----------
    run_index : int
        0 for physical movement anchor, 1-4 for MI runs.
    marker_outlet : pylsl.StreamOutlet
        Marker stream (``Task_Markers``). ``push_sample([label])`` at each
        event. LabRecorder saves these alongside the EEG for offline analysis.
    window : psychopy.visual.Window
        Pre-created window. Caller is responsible for its lifecycle.
    config : dict
        Parsed ``configs/task.yaml``.
    """
    raise NotImplementedError(
        "Stub. Implement following reference/simple_bci/task.py::Paradigm: "
        "build a balanced shuffled trial list, iterate, and on each trial "
        "flip frames for the fixation (3s) → beep (1s before cue) → cue (4s) "
        "→ ITI (random 2.5-4.5s) sequence, pushing an LSL marker at t=0."
    )


def ms_to_frames(ms: float, refresh_rate_hz: float) -> int:
    """Number of monitor frames closest to ``ms`` at the given refresh rate.

    PsychoPy paradigms always step in frame counts, never wall-clock ``time.sleep``.
    See reference/simple_bci/task.py::MsToFrames for the original.
    """
    raise NotImplementedError("Implement: round(ms * refresh_rate_hz / 1000.0).")
