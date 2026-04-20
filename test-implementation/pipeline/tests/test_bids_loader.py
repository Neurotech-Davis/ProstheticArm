"""Placeholder tests for the BIDS loader.

Marked as skipped until ``bids_loader`` is implemented. Reminder to the future
implementer: real tests should load sub-02 run-1, verify the Raw has 15 EEG
channels at 125 Hz and non-empty annotations with at least one MI + one Rest.
"""

from __future__ import annotations

import pytest

pytest.skip("bids_loader not yet implemented", allow_module_level=True)
