"""Placeholder tests for filter + epoch. Implement alongside preprocessing.

When implemented, cover:
  - bandpass attenuates a 50 Hz sinusoid by >20 dB
  - epoch count equals events_from_annotations count (no silent drops)
  - epoch length in samples == round((tmax - tmin) * sfreq)
"""

from __future__ import annotations

import pytest

pytest.skip("preprocessing not yet implemented", allow_module_level=True)
