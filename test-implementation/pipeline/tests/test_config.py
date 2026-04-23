"""Tests for config loading.

Light sanity checks — the YAML files should parse and carry the keys that
downstream code depends on. Catches typos in configs/*.yaml at test time
instead of at the start of a 10-fold LOSO run.
"""

from __future__ import annotations

from bci_grasp.config import load_config


def test_load_all_configs():
    """Every *.yaml in configs/ parses and lands under its filename stem."""
    all_cfg = load_config()
    for required in ("data", "preprocessing", "features", "model", "deployment", "task"):
        assert required in all_cfg, f"missing {required}.yaml"


def test_data_config_shape():
    cfg = load_config("data")
    assert "bids_root" in cfg
    assert len(cfg["subjects"]) == 10  # 10 valid subjects in ds003810
    assert cfg["sampling_rate_hz"] == 125
    assert set(cfg["deployment_channels"]) == {"C3", "Cz", "C4", "F3", "P3", "Pz"}
    assert cfg["event_id"] == {"MI": 7, "Rest": 9}


def test_preprocessing_band_is_sensible():
    cfg = load_config("preprocessing")
    l = cfg["bandpass"]["l_freq"]
    h = cfg["bandpass"]["h_freq"]
    assert 0 < l < h
    # Epoch window must be positive-length and within the 4s task cue.
    assert 0 <= cfg["epoch"]["tmin"] < cfg["epoch"]["tmax"] <= 4.0


def test_task_trial_counts_balanced():
    cfg = load_config("task")
    assert cfg["trials_per_run"]["mi"] == cfg["trials_per_run"]["rest"] == 20
