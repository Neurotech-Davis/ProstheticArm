"""LSL inlet for live EEG.

Scaffold only. Follows the same pattern as reference/simple_bci/backend.py:
resolve an LSL stream by name, open an inlet, and ``pull_sample`` in a loop.

Deployment source: OpenBCI Cyton + OpenBCI GUI publishes an "EEG" LSL stream.
For offline development, any LSL-capable mock source works (pylsl examples
ship with one).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pylsl


def resolve_eeg_inlet(name: str, timeout_s: float = 5.0) -> "pylsl.StreamInlet":
    """Block until an LSL stream named ``name`` is found, then open an inlet.

    Parameters
    ----------
    name : str
        Stream name (set in the OpenBCI GUI or source script).
    timeout_s : float
        Resolve timeout. Raises ``RuntimeError`` if no stream appears in time.

    Returns
    -------
    pylsl.StreamInlet
        Inlet with ``recover=False`` so a lost stream doesn't silently block
        the inference loop.
    """
    raise NotImplementedError(
        "Implement using pylsl.resolve_byprop('name', name, timeout=timeout_s) "
        "→ pylsl.StreamInlet(info[0], recover=False)."
    )
