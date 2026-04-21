"""Stage 3: cross-validated evaluation of the CSP → LDA pipeline.

Two schemes:
  - ``loso``   (primary)  — leave-one-subject-out, honest generalization.
  - ``kfold``  (secondary) — StratifiedKFold pooled across subjects, upper bound.

Writes:
  - ``reports/metrics/<run_id>.json``  — per-fold + summary metrics
  - ``reports/figures/<run_id>_confusion.png``

Usage
-----
    conda activate psychopy_env
    python scripts/03_evaluate.py --scheme loso
    python scripts/03_evaluate.py --scheme kfold
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bci_grasp.config import PIPELINE_ROOT, load_config
from bci_grasp.evaluation.cv import kfold_splitter, loso_splitter
from bci_grasp.evaluation.metrics import accuracy, confusion, roc_auc
from bci_grasp.models.lda_pipeline import build_lda_pipeline

# Reuse the loader from 02_train to avoid drift.
import sys

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

_train_mod = import_module("02_train")  # type: ignore[assignment]
load_all_epochs = _train_mod.load_all_epochs

log = logging.getLogger("evaluate")


def run_cv(X, y, groups, splitter, kfold_needs_groups: bool):
    """Iterate the splitter and return per-fold metrics + aggregated confusion."""
    per_fold = []
    conf_total = np.zeros((2, 2), dtype=int)

    cfg_all = load_config()
    cfg_feat = cfg_all["features"]["csp"]
    cfg_model = cfg_all["model"]["model"]

    split_iter = (
        splitter.split(X, y, groups) if kfold_needs_groups else splitter.split(X, y)
    )

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        pipe = build_lda_pipeline(
            csp_n_components=cfg_feat["n_components"],
            csp_reg=cfg_feat.get("reg"),
            lda_solver=cfg_model["params"]["solver"],
            lda_shrinkage=cfg_model["params"]["shrinkage"],
        )
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        # decision_function is signed — pass to roc_auc as the positive-class score.
        y_prob = pipe.decision_function(X[test_idx])

        acc = accuracy(y[test_idx], y_pred)
        auc = roc_auc(y[test_idx], y_prob)
        conf = confusion(y[test_idx], y_pred)
        conf_total += conf

        held_out = str(np.unique(groups[test_idx])) if groups is not None else ""
        per_fold.append(
            {"fold": fold_idx, "held_out": held_out, "accuracy": acc, "roc_auc": auc}
        )
        log.info("fold %d%s: acc=%.3f auc=%.3f", fold_idx, f" (held_out={held_out})" if held_out else "", acc, auc)

    return per_fold, conf_total


def save_confusion_figure(conf: np.ndarray, out_path: Path) -> None:
    """Save a simple labelled confusion matrix PNG."""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(conf, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                    color="white" if conf[i, j] > conf.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Rest", "MI"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Rest", "MI"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=["loso", "kfold"], default="loso")
    args = parser.parse_args()

    cfg = load_config("data")
    X, y, subs = load_all_epochs(cfg["subjects"])
    groups = np.array(subs)
    log.info(
        "loaded %d trials from %d subjects", len(y), len(set(subs))
    )

    if args.scheme == "loso":
        if len(set(subs)) < 2:
            log.warning(
                "LOSO requested but only %d subject(s) available — falling back to k-fold. "
                "Fetch more subjects with `datalad get sub-XX` and rerun 01_preprocess.",
                len(set(subs)),
            )
            splitter = kfold_splitter()
            kfold_needs_groups = False
            effective_scheme = "kfold_fallback"
        else:
            splitter = loso_splitter()
            kfold_needs_groups = True
            effective_scheme = "loso"
    else:
        splitter = kfold_splitter()
        kfold_needs_groups = False
        effective_scheme = "kfold"

    per_fold, conf_total = run_cv(X, y, groups, splitter, kfold_needs_groups)
    accs = np.array([p["accuracy"] for p in per_fold])
    aucs = np.array([p["roc_auc"] for p in per_fold])
    summary = {
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "roc_auc_mean": float(np.nanmean(aucs)),
        "roc_auc_std": float(np.nanstd(aucs)),
        "n_folds": int(len(per_fold)),
        "scheme": effective_scheme,
    }
    log.info(
        "summary: acc=%.3f±%.3f  auc=%.3f±%.3f  (%d folds)",
        summary["accuracy_mean"], summary["accuracy_std"],
        summary["roc_auc_mean"], summary["roc_auc_std"], summary["n_folds"],
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_lda_{effective_scheme}"
    reports = PIPELINE_ROOT / "reports"
    (reports / "metrics").mkdir(parents=True, exist_ok=True)
    (reports / "metrics" / f"{run_id}.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "per_fold": per_fold,
                "confusion_total": conf_total.tolist(),
            },
            indent=2,
        )
    )
    save_confusion_figure(conf_total, reports / "figures" / f"{run_id}_confusion.png")
    log.info("wrote reports/metrics/%s.json + reports/figures/%s_confusion.png", run_id, run_id)


if __name__ == "__main__":
    main()
