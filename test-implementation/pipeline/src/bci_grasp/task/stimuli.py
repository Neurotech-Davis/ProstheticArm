"""Visual + auditory stimulus objects for the paradigm.

Factory functions that build configured PsychoPy objects from
``configs/task.yaml``. Keeping them as factories (rather than module-level
instances) means the GUI module doesn't import PsychoPy until the user
actually runs the task — useful on machines without a display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psychopy.sound
    import psychopy.visual


def make_fixation(window: "psychopy.visual.Window", cfg: dict) -> "psychopy.visual.ShapeStim":
    """White cross at window center. Always drawn during preparation/break."""
    raise NotImplementedError(
        "Implement: psychopy.visual.ShapeStim(window, vertices='cross', size=cfg['size_px'], "
        "fillColor=cfg['color'], lineColor=cfg['color'], units='pix', pos=[0, 0])."
    )


def make_mi_arrow(
    window: "psychopy.visual.Window", cfg: dict
) -> "psychopy.visual.ShapeStim":
    """Red right-pointing arrow.

    Drawn for the 4 s task window on MI trials only. Rest trials draw nothing
    (only the fixation cross stays on).
    """
    raise NotImplementedError(
        "Implement: build arrow vertices pointing right, return ShapeStim with "
        "cfg['color'] (red) and cfg['size_px']."
    )


def make_beep(cfg: dict) -> "psychopy.sound.Sound":
    """Audible warning cue played at t=-1 s."""
    raise NotImplementedError(
        "Implement: psychopy.sound.Sound(value=cfg['freq_hz'], secs=cfg['duration_s'])."
    )
