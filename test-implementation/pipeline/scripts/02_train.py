"""Stage 2: fit CSP + LDA on the preprocessed epochs.

Reads the epoched .fif files from derivatives/, concatenates across subjects,
fits an sklearn Pipeline (CSP → LDA), and saves to ``models/<run_id>.joblib``
with a sidecar JSON (config snapshot + train metrics + git SHA).

Usage:
    conda activate psychopy_env
    python scripts/02_train.py
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Stub. Implement: load all subjects' epochs from derivatives/ → X (n_epochs, "
        "n_channels, n_times), y (0/1), groups (subject) → build_lda_pipeline → fit → "
        "save_model."
    )


if __name__ == "__main__":
    main()
