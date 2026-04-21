"""Tests for the ds003810 BIDS loader.

Requires sub-02 to have been fetched via ``datalad get sub-02`` — tests
skip cleanly otherwise so CI (with no data) still passes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from bci_grasp.config import resolve_repo_path
from bci_grasp.data.bids_loader import GDF_TO_LABEL, edf_path, load_run
from bci_grasp.data.channel_select import pick_deployment_channels

BIDS_ROOT = resolve_repo_path("test-implementation/ds003810")
SUB = "02"
RUN = 1

_edf = edf_path(BIDS_ROOT, SUB, RUN)
_data_available = _edf.exists() and not _edf.is_symlink() or (_edf.exists() and _edf.resolve().exists())

pytestmark = pytest.mark.skipif(
    not _data_available, reason=f"EDF not present ({_edf}). Run `datalad get sub-{SUB}`."
)


def test_load_run_shape_and_rate():
    raw = load_run(BIDS_ROOT, SUB, RUN)
    assert raw.info["sfreq"] == 125.0
    assert len(raw.ch_names) == 15


def test_annotations_renamed_to_mi_and_rest():
    """GDF_Right/GDF_Tongue should be renamed to MI/Rest; originals should not survive."""
    raw = load_run(BIDS_ROOT, SUB, RUN)
    descs = Counter(str(d) for d in raw.annotations.description)
    assert descs["MI"] == 20
    assert descs["Rest"] == 20
    for raw_label in GDF_TO_LABEL:
        assert raw_label not in descs, f"raw label {raw_label} leaked through rename"


def test_data_is_in_volts_after_unit_fix():
    """After the uV→V rescale, epoched peak-to-peak should be well under 1 V."""
    import numpy as np

    raw = load_run(BIDS_ROOT, SUB, RUN)
    max_abs = float(np.abs(raw.get_data()).max())
    # Tens of microvolts translates to ~1e-4 V; should be << 1 V.
    assert max_abs < 1e-2, f"Data still looks unscaled: max |x| = {max_abs}"


def test_channel_select_subset_and_order():
    raw = load_run(BIDS_ROOT, SUB, RUN)
    picked = pick_deployment_channels(raw, ["C3", "Cz", "C4", "F3", "P3", "Pz"])
    assert picked.ch_names == ["C3", "Cz", "C4", "F3", "P3", "Pz"]


def test_channel_select_strict_raises_on_missing():
    raw = load_run(BIDS_ROOT, SUB, RUN)
    with pytest.raises(ValueError, match="missing"):
        pick_deployment_channels(raw, ["C3", "NOT_A_CHANNEL"], strict=True)
