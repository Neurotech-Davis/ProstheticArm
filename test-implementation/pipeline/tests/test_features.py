"""Placeholder tests for CSP + bandpower. Implement alongside features.

When implemented, cover:
  - CSP reduces (n_epochs, n_ch, n_times) → (n_epochs, n_components)
  - bandpower output shape == (n_epochs, n_channels * n_bands)
  - on synthetic data where class 1 has strong 10 Hz bursts on C3 and class 0 doesn't,
    CSP+LDA achieves >90% 5-fold accuracy (smoke test).
"""

from __future__ import annotations

import pytest

pytest.skip("features not yet implemented", allow_module_level=True)
