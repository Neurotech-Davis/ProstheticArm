"""Stage 2: fit CSP + LDA on preprocessed epochs, save model + sidecar.

Loads all per-subject ``*_epo.fif`` from ``derivatives/bci-grasp-vs-rest/``
(produced by ``01_preprocess.py``), fits the CSP→LDA pipeline on every
available trial, and writes:

    models/<run_id>.joblib        # fitted pipeline
    models/<run_id>.json          # config snapshot + metrics + git SHA

This script fits on ALL available data — it does not cross-validate. Use
``03_evaluate.py`` for LOSO / k-fold accuracy estimates.

Usage
-----
    conda activate psychopy_env
    python scripts/02_train.py
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import mne
import numpy as np

from bci_grasp.config import PIPELINE_ROOT, load_config
from bci_grasp.models.base import save_model
from bci_grasp.models.lda_pipeline import build_lda_pipeline

log = logging.getLogger("train")


def derivatives_root() -> Path:
    return PIPELINE_ROOT / "derivatives" / "bci-grasp-vs-rest"


def load_all_epochs(subjects: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Concatenate every subject's epochs into one (X, y, subject_per_trial)."""
    Xs, ys, subs = [], [], []
    for sub in subjects:
        fif = derivatives_root() / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-MIvsRest_epo.fif"
        if not fif.exists():
            log.warning("sub-%s: no preprocessed epochs — skipping (run 01_preprocess first)", sub)
            continue
        ep = mne.read_epochs(fif, verbose="ERROR")
        Xs.append(ep.get_data())
        ys.append(ep.events[:, -1])
        subs.extend([sub] * len(ep))
    if not Xs:
        raise RuntimeError("No preprocessed epochs found. Run 01_preprocess.py first.")
    return np.concatenate(Xs, 0), np.concatenate(ys, 0), subs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="Optional tag appended to run_id.")
    args = parser.parse_args()

    cfg_all = load_config()
    cfg_data = cfg_all["data"]
    cfg_model = cfg_all["model"]

    X, y, subs = load_all_epochs(cfg_data["subjects"])
    log.info(
        "loaded %d trials from %d subjects (MI=%d, Rest=%d)",
        len(y),
        len(set(subs)),
        int((y == 1).sum()),
        int((y == 0).sum()),
    )

    pipe = build_lda_pipeline(
        csp_n_components=cfg_all["features"]["csp"]["n_components"],
        csp_reg=cfg_all["features"]["csp"].get("reg"),
        lda_solver=cfg_model["model"]["params"]["solver"],
        lda_shrinkage=cfg_model["model"]["params"]["shrinkage"],
    )
    pipe.fit(X, y)
    train_acc = float((pipe.predict(X) == y).mean())
    log.info("train-set accuracy (optimistic, no CV): %.3f", train_acc)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{cfg_model['model']['name']}"
    if args.tag:
        run_id += f"_{args.tag}"

    out_path = PIPELINE_ROOT / "models" / f"{run_id}.joblib"
    sidecar = save_model(
        pipe,
        out_path,
        config=cfg_all,
        metrics={"train_accuracy": train_acc, "n_trials": int(len(y)), "n_subjects": len(set(subs))},
    )
    log.info("saved %s (+ %s)", out_path.name, sidecar.name)


if __name__ == "__main__":
    main()
