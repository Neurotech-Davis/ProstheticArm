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

## Training the classifier

Prerequisite: fetch at least one subject of ds003810 (see `../claudecontext.md`
for `datalad get` setup). With `sub-02` on disk:

```bash
conda activate psychopy_env
python scripts/01_preprocess.py --subject 02     # BIDS → derivatives/*_epo.fif
python scripts/02_train.py                       # CSP → LDA → models/<run_id>.joblib + .json
python scripts/03_evaluate.py --scheme loso      # LOSO (falls back to k-fold for <2 subjects)
```

Outputs:
- `derivatives/bci-grasp-vs-rest/sub-XX/eeg/sub-XX_task-MIvsRest_epo.fif` — preprocessed epochs per subject
- `models/<run_id>.joblib` + sidecar `.json` (config snapshot, metrics, git SHA)
- `reports/metrics/<run_id>.json` + `reports/figures/<run_id>_confusion.png`

For full LOSO, `datalad get sub-XX` each remaining subject and rerun `01_preprocess.py`
(no `--subject` flag → all).

## Testing the Arduino link

Manually exercise the Python→Arduino serial bridge without the classifier
using the PsychoPy toggle GUI:

```bash
conda activate psychopy_env
cd test-implementation/pipeline

# Without Arduino plugged in (prints what it would send):
python scripts/test_arduino_gui.py --port MOCK

# With Arduino (upload grasp_controller.ino first, then):
python scripts/test_arduino_gui.py --port /dev/cu.usbmodem1101
```

Click the big button to toggle REST ↔ GRASP. The on-board LED should follow;
servos will follow once you uncomment the PCA9685 lines in the `.ino`.
ESC or the STOP button quits.

> **⚠️ macOS serial port names vary per USB port.** The suffix (`1101`, `101`,
> `14101`, `14201`, …) is assigned by macOS based on which physical USB port
> you plug into, not by the Arduino. If you get
> `OS error: cannot open port /dev/cu.usbmodemXXXX: No such file or directory`
> either in the Arduino IDE or in the Python GUI, run:
>
> ```bash
> ls /dev/cu.usbmodem*
> ```
>
> Use whatever it prints as `--port`. Switching USB ports (or rebooting) can
> change the suffix — always confirm after plugging in. Prefer `/dev/cu.*`
> over `/dev/tty.*`; the `tty.*` variant blocks on open and is the wrong
> device class for Arduino programming / pyserial on macOS.

## Status (first pass)

| Module | Status |
|---|---|
| `data/` (BIDS loader + channel select) | **implemented** |
| `preprocessing/` (filter + epoch) | **implemented** |
| `artifacts/reject.py` | **implemented** |
| `features/csp.py` | **implemented** |
| `models/` (LDA pipeline + save/load) | **implemented** |
| `evaluation/` (LOSO + k-fold + metrics) | **implemented** |
| `scripts/01_preprocess.py`, `02_train.py`, `03_evaluate.py` | **implemented** |
| `deployment/arduino_serial.py` (single-byte ASCII link) | **implemented** |
| `arduino/grasp_controller/grasp_controller.ino` | **implemented** (LED blinks on commands; PCA9685 servo lines ready to uncomment) |
| `scripts/test_arduino_gui.py` (toggle GUI) | **implemented** |
| `deployment/protocol.py` + tests (framed protocol — reference only) | **implemented, unused** |
| `realtime/` (LSL inlet + sliding inference) | stub (wire up after LOSO validation) |
| `task/` (PsychoPy stimulus GUI) | stub (for future own-data collection) |

## Conventions

- **Config-driven.** No hard-coded channels, bands, or paths in code. Read from `configs/*.yaml`.
- **BIDS-first.** Read `ds003810/` via `mne-bids`. Write processed EEG under `derivatives/bci-grasp-vs-rest/`.
- **Channel subset.** 6 channels (C3, Cz, C4, F3, P3, Pz) to match planned OpenBCI deployment.
- **Event mapping.** Dataset code `7` → MI (class 1); `9` → Rest (class 0). GDF names in raw EDF (`OVTK_GDF_Right` / `OVTK_GDF_Tongue`) are misleading — trust the BIDS events JSON.
- **Docs.** Module and function docstrings; inline comments on non-obvious BCI choices (filter band, epoch window, CSP component count).
