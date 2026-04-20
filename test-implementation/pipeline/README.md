# pipeline/ — BCI grasp-vs-rest prototype

Offline-first Motor Imagery BCI that classifies **right-hand grasp (MI) vs rest**
from EEG (OpenNeuro `ds003810`) and commands an Arduino-driven prosthetic arm
over serial. See `../claudecontext.md` for the full project context.

## Layout

```
pipeline/
├── configs/                 YAML configs — no magic numbers in code
├── src/bci_grasp/           importable package
│   ├── data/                BIDS loading + channel selection
│   ├── preprocessing/       filter, epoch
│   ├── artifacts/           trial rejection
│   ├── features/            CSP, bandpower
│   ├── models/              LDA pipeline (first pass)
│   ├── evaluation/          LOSO + k-fold CV, metrics
│   ├── realtime/            LSL inlet + sliding-window inference (stubs)
│   ├── deployment/          Python→Arduino serial protocol
│   └── task/                PsychoPy stimulus GUI (stubs)
├── scripts/                 CLI entrypoints
├── arduino/grasp_controller grasp_controller.ino (PCA9685 state machine)
├── notebooks/               exploratory only
├── tests/                   pytest
├── derivatives/             BIDS-Derivatives output root (processed EEG)
├── models/                  trained model artifacts (joblib)
└── reports/                 figures + metrics
```

## Quickstart

All installs and runs happen inside the `psychopy_env` conda env.

```bash
conda activate psychopy_env
pip install -e .                      # install this package in editable mode
pip install -r requirements.txt       # fill in missing deps (mne-bids, sklearn, pytest)
pytest tests/                         # sanity check
```

## Status (first pass)

| Module | Status |
|---|---|
| `deployment/protocol.py` + tests | **implemented** (binary serial frame encode/decode + CRC8) |
| everything else | **stubs** — signatures + docstrings, `NotImplementedError` bodies |

Stubs are filled in incrementally. Start with `scripts/02_train.py` once
`data/bids_loader.py`, `preprocessing/`, `features/csp.py`, and
`models/lda_pipeline.py` are implemented.

## Conventions

- **Config-driven.** No hard-coded channels, bands, or paths in code. Read from `configs/*.yaml`.
- **BIDS-first.** Read `ds003810/` via `mne-bids`. Write processed EEG under `derivatives/bci-grasp-vs-rest/`.
- **Channel subset.** 6 channels (C3, Cz, C4, F3, P3, Pz) to match planned OpenBCI deployment.
- **Event mapping.** Dataset code `7` → MI (class 1); `9` → Rest (class 0). GDF names in raw EDF (`OVTK_GDF_Right` / `OVTK_GDF_Tongue`) are misleading — trust the BIDS events JSON.
- **Docs.** Module and function docstrings; inline comments on non-obvious BCI choices (filter band, epoch window, CSP component count).
