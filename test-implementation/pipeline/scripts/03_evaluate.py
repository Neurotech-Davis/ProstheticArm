"""Stage 3: cross-validated evaluation (LOSO primary, k-fold secondary).

Fits the pipeline per fold and reports accuracy + ROC-AUC + confusion matrix.
Writes a JSON summary to ``reports/metrics/<run_id>.json`` and a confusion
matrix figure to ``reports/figures/<run_id>_confusion.png``.

Usage:
    conda activate psychopy_env
    python scripts/03_evaluate.py --scheme loso
    python scripts/03_evaluate.py --scheme kfold
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=["loso", "kfold"], default="loso")
    args = parser.parse_args()
    _ = args

    raise NotImplementedError(
        "Stub. Implement: load epochs+groups → pick splitter (loso_splits / "
        "kfold_splits) → for each fold: fit, predict, compute metrics → "
        "aggregate → save JSON + confusion-matrix figure."
    )


if __name__ == "__main__":
    main()
