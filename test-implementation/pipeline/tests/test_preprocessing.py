"""Tests for preprocessing (filter + epoch).

Synthetic-data tests run always. A live-data test against sub-02 runs when
the EDF is fetched.
"""

from __future__ import annotations

import numpy as np
import pytest

from bci_grasp.config import resolve_repo_path
from bci_grasp.data.bids_loader import edf_path, load_run
from bci_grasp.data.channel_select import pick_deployment_channels
from bci_grasp.preprocessing.epoch import epoch_cue_locked
from bci_grasp.preprocessing.filter import apply_bandpass

BIDS_ROOT = resolve_repo_path("test-implementation/ds003810")
SUB = "02"
RUN = 1
_live_data = edf_path(BIDS_ROOT, SUB, RUN).resolve().exists()


def _make_synthetic_raw():
    """200 Hz sfreq, 10 s of sum(10 Hz + 50 Hz) sines on a single 'EEG' channel."""
    import mne

    sfreq = 200.0
    t = np.arange(0, 10, 1.0 / sfreq)
    # 1 uV amplitude at each freq — units stored as volts in MNE.
    data = 1e-6 * (np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t))
    info = mne.create_info(ch_names=["EEG"], sfreq=sfreq, ch_types=["eeg"])
    return mne.io.RawArray(data[np.newaxis, :], info, verbose="ERROR")


def test_bandpass_attenuates_50_hz_sinusoid():
    """8-30 Hz bandpass should drop the 50 Hz component by >20 dB relative to the 10 Hz one."""
    raw = _make_synthetic_raw()
    apply_bandpass(raw, l_freq=8.0, h_freq=30.0)

    x = raw.get_data()[0]
    sfreq = raw.info["sfreq"]
    freqs = np.fft.rfftfreq(len(x), d=1 / sfreq)
    amps = np.abs(np.fft.rfft(x))

    a10 = amps[np.argmin(np.abs(freqs - 10))]
    a50 = amps[np.argmin(np.abs(freqs - 50))]
    assert a10 > 0
    # 20 dB = 10x amplitude ratio.
    assert a10 / max(a50, 1e-12) > 10, f"50 Hz not attenuated: ratio {a10 / a50}"


@pytest.mark.skipif(not _live_data, reason="sub-02 EDF not fetched")
def test_epoch_count_matches_annotations():
    """Number of epochs should equal the number of MI + Rest annotations for the run."""
    raw = load_run(BIDS_ROOT, SUB, RUN)
    raw = pick_deployment_channels(raw, ["C3", "Cz", "C4", "F3", "P3", "Pz"])
    apply_bandpass(raw, 8.0, 30.0)
    ep = epoch_cue_locked(raw, tmin=0.5, tmax=2.5)
    from collections import Counter

    ann = Counter(str(d) for d in raw.annotations.description)
    expected = ann["MI"] + ann["Rest"]
    assert len(ep) == expected


@pytest.mark.skipif(not _live_data, reason="sub-02 EDF not fetched")
def test_epoch_sample_count_matches_window_at_125hz():
    """(tmax - tmin) * sfreq + 1 samples (MNE tmax is inclusive)."""
    raw = load_run(BIDS_ROOT, SUB, RUN)
    raw = pick_deployment_channels(raw, ["C3", "Cz", "C4", "F3", "P3", "Pz"])
    apply_bandpass(raw, 8.0, 30.0)
    ep = epoch_cue_locked(raw, tmin=0.5, tmax=2.5)
    assert ep.get_data().shape == (40, 6, int((2.5 - 0.5) * 125) + 1)
