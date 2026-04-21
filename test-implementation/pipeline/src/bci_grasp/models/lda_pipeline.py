"""CSP → log-variance → LDA pipeline (first-pass classifier).

Why LDA:
  - CSP log-variance features are near-Gaussian per class, which is exactly
    LDA's assumption.
  - No hyperparameters to tune (beyond shrinkage, which has a closed-form
    estimate via ``shrinkage="auto"``) → less risk of overfitting on a tiny
    10-subject LOSO scheme.
  - Fast to train and evaluate — useful when iterating on preprocessing.

The whole thing is an ``sklearn.pipeline.Pipeline`` so it composes with
``cross_val_score`` and ``LeaveOneGroupOut``.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from bci_grasp.features.csp import build_csp


def build_lda_pipeline(
    csp_n_components: int = 4,
    csp_reg: str | None = None,
    lda_solver: str = "lsqr",
    lda_shrinkage: str | float | None = "auto",
) -> Pipeline:
    """Construct the CSP → LDA sklearn Pipeline.

    Parameters
    ----------
    csp_n_components : int
    csp_reg : str or None
        See ``features.csp.build_csp``.
    lda_solver : {"lsqr", "eigen", "svd"}
        "lsqr" + shrinkage is the standard BCI recipe.
    lda_shrinkage : "auto", float, or None
        "auto" uses Ledoit-Wolf. Only valid with lsqr/eigen solvers; ignored
        (must be None) for "svd".

    Returns
    -------
    sklearn.pipeline.Pipeline
        Named steps: ``"csp"``, ``"lda"``. Fit with ``(X, y)`` where X has
        shape ``(n_epochs, n_channels, n_times)`` and y is ``{0, 1}``.
    """
    lda_kwargs: dict = {"solver": lda_solver}
    if lda_solver in ("lsqr", "eigen"):
        lda_kwargs["shrinkage"] = lda_shrinkage
    # "svd" solver doesn't accept shrinkage at all — leave it out.

    return Pipeline(
        [
            ("csp", build_csp(n_components=csp_n_components, reg=csp_reg, log=True)),
            ("lda", LinearDiscriminantAnalysis(**lda_kwargs)),
        ]
    )
