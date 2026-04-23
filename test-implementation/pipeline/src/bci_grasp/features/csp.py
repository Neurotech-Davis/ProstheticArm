"""Common Spatial Patterns (CSP).

CSP finds linear combinations of channels (spatial filters) whose variance is
maximal for one class and minimal for the other — exactly the decomposition
you want for ERD-based motor-imagery classification, because ERD manifests
as a variance (band-power) difference between MI and Rest.

Classic binary CSP+LDA pipeline:

    1. Bandpass filter (done in preprocessing).
    2. Epoch around the cue (done in preprocessing).
    3. Fit CSP on training epochs → spatial filter matrix W.
    4. Apply W → log-variance of each component → n_components features/trial.
    5. LDA on those features.

We reuse ``mne.decoding.CSP`` rather than reimplementing the generalized
eigenvalue problem. The factory below just pins the defaults we care about
and returns an object that drops straight into ``sklearn.pipeline.Pipeline``.
"""

from __future__ import annotations

from mne.decoding import CSP


def build_csp(
    n_components: int = 4, reg: str | None = None, log: bool = True
) -> CSP:
    """Construct a configured ``mne.decoding.CSP`` transformer.

    Parameters
    ----------
    n_components : int
        Number of spatial filters kept. With 6 input channels, 4 is a
        reasonable default (2 per class "direction"). Drop to 2 if LOSO
        shows overfitting.
    reg : {None, "ledoit_wolf", "oas"} or float
        Covariance regularization. ``None`` is fine when trial count per
        subject is high; "ledoit_wolf" helps with few trials / LOSO.
    log : bool
        If True, returns log-variance of filtered signal as features
        (standard). If False, returns z-scored variance — sometimes slightly
        better but less interpretable.

    Returns
    -------
    mne.decoding.CSP
        Ready to ``fit`` on ``(n_epochs, n_channels, n_times)`` data with
        integer labels in ``y``.
    """
    return CSP(
        n_components=n_components,
        reg=reg,
        log=log,
        cov_est="concat",
    )
