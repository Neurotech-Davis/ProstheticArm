# claudecontext.md

Primer for Claude (or a new collaborator) working inside `test-implementation/`.
Keep this concise. Update when goals, layout, or decisions change.

---

## Project goal


Build an **offline Motor Imagery BCI** that classifies **right-hand grasp (MI) vs rest** from EEG, then issues a command over serial to an Arduino driving a servo-based prosthetic arm.

Neurotech 2026 project. Three currently-disconnected pieces are being integrated into one pipeline:

1. `ds003810/` — OpenNeuro BIDS dataset used for training/validation.
2. `reference/simple_bci/` — LSL + PsychoPy real-time skeleton (ollie-d, 2023). Structural template only.
3. `reference/Arduino/` — PCA9685 servo sweep reference. Hardware check only.

The missing glue is: a trained classifier, a real-time inference loop, and a Python→Arduino serial bridge.

---

## Directory layout

```
test-implementation/
├── claudecontext.md                   ← this file
├── ds003810/                          raw BIDS dataset (read-only input)
│   ├── sub-{02,03,04,05,06,07,08,09,10,12}/eeg/*.edf
│   ├── task-MIvsRest_{events,channels}.json
│   ├── task-MIvsRest_channels.tsv
│   ├── participants.tsv
│   └── Code/Demo_datause.ipynb        git-annex symlink — NOT downloaded
├── reference/                         read-only reference code, do NOT edit
│   ├── simple_bci/
│   │   ├── task.py                    PsychoPy paradigm + LSL markers
│   │   ├── backend.py                 LSL consumer stub (random classifier)
│   │   ├── eeg_utils.py               XDF load + epoch helpers
│   │   └── data/*.xdf
│   └── Arduino/reference_motor_control.cpp    PCA9685 sweep demo
└── pipeline/                          ← our prototype (to be scaffolded)
```

---

## Dataset key specs (ds003810)

- **Paper:** Peterson et al. 2020, Heliyon. Low-cost consumer-grade BCI, MI-vs-Rest.
- **Hardware:** OpenBCI Cyton+Daisy, Electro-Cap System II (wet tin electrodes).
- **Subjects:** 12 recorded, **10 valid** — folders sub-02, 03, 04, 05, 06, 07, 08, 09, 10, 12. All right-hand dominant, ages 20–30.
- **Channels (15, 10–20):** Fz, F3, F4, F7, F8, Cz, C3, C4, T3, T4, Pz, P3, P4, T5, T6. **REF** = A1 (left earlobe), **GND** = A2 (right earlobe).
- **Sampling rate:** 125 Hz.
- **Filters:** online 0.5–45 Hz Butterworth (3rd order). Powerline 50 Hz (Argentina).
- **Runs per subject:** 5. RUN 0 = physical grasping (anchor). RUN 1–4 = pure MI.
- **Trials per run:** 40 (20 MI, 20 Rest).
- **Event codes (in events JSON):** `7` = MI (raw EDF label `OVTK_GDF_Right`), `9` = Rest (raw EDF label `OVTK_GDF_Tongue`). The GDF names are misleading — trust the JSON mapping.
- **Markers** land at cue onset (t=0s of the 4s task window).
- **EMG (Myoware)** was used for artifact rejection in the paper but **no EMG files are present** in this local copy — treat as EEG-only.

---

## Paradigm (for our stimulus GUI and data interpretation)

Modified Graz protocol, binary MI vs Rest:

- t = −3 s: fixation cross (preparation)
- t = −1 s: audible beep (warning)
- t =  0 s: task onset, 4 s window. **MI** = red arrow pointing right; **Rest** = no arrow
- Each run contains exactly 20 MI + 20 Rest trials (40 total), with trial order randomized (balanced sequence, shuffled — not independent random draws per trial)
- Inter-trial interval: random **2.5–4.5 s**
- Subject's dominant hand is placed **inside a cardboard box** to force kinesthetic (not visual) imagery
- LSL marker is pushed at **t=0s** (matches "time mark stamps" in dataset)
- RUN 0 = physical movement; RUN 1–4 = pure MI

---

## Pipeline plan (to be built under `pipeline/`)

**Layout standards.** `pipeline/` blends three conventions:
- **Cookiecutter Data Science** — overall shape (`src/<package>/`, `configs/`, `scripts/`, `notebooks/`, `tests/`, `models/`, `reports/`).
- **BIDS-Derivatives** — processed EEG outputs under `derivatives/bci-grasp-vs-rest/sub-XX/eeg/` with a `dataset_description.json` at the derivative root.
- **MNE-Python / MOABB** — BCI stack conventions (CSP → LDA `sklearn.Pipeline`, LOSO CV via `LeaveOneGroupOut`).

High-level stages and defaults. Flexibility noted — revisit each choice with evidence.

| Stage | Default | Notes |
|---|---|---|
| Load | MNE-BIDS `read_raw_bids` | map event 7→MI(1), 9→Rest(0) |
| Channel select | 6 deployment channels: **C3, Cz, C4, F3, P3, Pz** | all present in the 15-ch set |
| Preprocessing | bandpass 8–30 Hz, optional 50 Hz notch | task-band on top of the hardware filter |
| Epoch | cue-locked window **0.5–2.5 s** post-cue | 250 samples at 125 Hz |
| Artifact rejection | amplitude threshold; autoreject optional | EMG unavailable |
| Features | **CSP** → log-variance of top components | bandpower (μ 8–13, β 13–30) as fallback |
| Classifier | **LDA** (first pass) | SVM / EEGNet later |
| Evaluation | **LOSO** CV (10 subjects) + k-fold | accuracy + ROC-AUC + confusion matrix |
| Realtime | **scaffold only** (LSL inlet + sliding predict) | offline is the working focus |
| Deployment | Python→Arduino over serial, **single ASCII byte** (`'0'`=rest, `'1'`=grasp) | see below |

### Arduino serial protocol

Single ASCII byte per state change:

```
Python → Arduino:
    '1' (0x31)   grasp   → servos to SERVO_MAX
    '0' (0x30)   rest    → servos to SERVO_MIN
    anything else        → ignored (resilient to startup noise)
```

Send on transition, not every classifier tick. Apply hysteresis
(`configs/deployment.yaml::decision_threshold`, `min_dwell_s`) in Python
before calling `link.send(...)` — the classifier's raw output will
chatter without it.

Implementation: `pipeline/src/bci_grasp/deployment/arduino_serial.py`
(pyserial wrapper) and `pipeline/arduino/grasp_controller/grasp_controller.ino`
(single-byte reader).

**Legacy framed protocol** (5-byte `[0xAA][0x55][CMD][ARG][CRC8]`) lives in
`pipeline/src/bci_grasp/deployment/protocol.py` + `pipeline/tests/test_protocol.py`.
Kept for reference / future use — not wired into the v1 path. Revisit only
when we add a third command (calibration, speed, telemetry back, etc.).

### Stimulus GUI (stubs only at first)

Mirrors `reference/simple_bci/task.py` structure (PsychoPy + LSL marker outlet). Graz paradigm constants in `pipeline/configs/task.yaml`. `refresh_rate` constant MUST match monitor refresh — carry over that warning.

---

## Working conventions

- **Conda env:** all Python installs must go into **`psychopy_env`**. Activate before `pip install` / `conda install`.
- **In-file documentation:** `pipeline/` code should be thoroughly documented — module/function docstrings, why-comments on non-obvious BCI choices (filter bands, epoch window, CSP, LDA). Not every line, but enough for a learner to read the code end-to-end.
- **Do not edit** `reference/` or `ds003810/` — both are read-only inputs.
- **BIDS derivatives:** processed EEG outputs land under `pipeline/derivatives/bci-grasp-vs-rest/sub-XX/eeg/` with `dataset_description.json` at the derivative root.
- **Dataset path:** read from `test-implementation/ds003810/` directly. Do not copy or re-symlink into `pipeline/`.

## Dataset setup (ds003810 is git-annex managed)

The ds003810 repo ships as a DataLad / git-annex dataset: every `*.edf` is a symlink into `.git/annex/objects/...`, and the actual bytes must be fetched explicitly. The dataset's git repo is kept in place (its nested `.git` is fine) — `test-implementation/ds003810/` is added to the repo's `.gitignore` so the outer git ignores it entirely (no submodule noise).

Required tooling:

- **`git-annex`** — system CLI (Haskell binary). Install via **Homebrew**: `brew install git-annex`. Not available on conda-forge for osx-arm64, so it's the one exception to the "everything into psychopy_env" rule.
- **`datalad`** — Python wrapper around git-annex. Install into `psychopy_env`: `pip install datalad`.

First-time setup:

```bash
# one-time, system
brew install git-annex

# into psychopy_env
conda activate psychopy_env
pip install datalad

# fetch data (per-subject to keep downloads small)
cd test-implementation/ds003810
datalad get sub-02               # single subject for dev, ~20 MB
# later, for full LOSO:
datalad get .
```

If the EDFs appear as broken symlinks (`ls -la` shows `../.git/annex/objects/...` and `file` reports "broken symbolic link"), the annex objects haven't been fetched yet — run `datalad get`.

---

## Open items / handoff notes

- `ds003810/Code/Demo_datause.ipynb` is a git-annex symlink and is not present locally — `git annex get` before reading.
- EEGNet is **not** part of the initial scaffold; add later if LDA proves insufficient.
- The stimulus GUI is initially for **offline data collection** (pair with LabRecorder → XDF). Driving the online classifier loop from the GUI is a later extension.
- No EMG in this copy of ds003810 — artifact rejection must rely on EEG-only methods.
