# notebooks/

Exploratory notebooks. Numbered for order, not for dependency — each should
be self-contained (load config + data at top, don't rely on state from a
prior notebook).

Planned:

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | Topomaps, PSDs, event counts per subject; sanity-check the raw EDFs before writing any filtering code. |
| 02 | `02_preprocessing_sanity.ipynb` | Overlay raw vs filtered traces; show the bandpass attenuates 50 Hz by the expected amount. |
| 03 | `03_csp_visualization.ipynb` | Fit CSP on one subject; plot spatial patterns to confirm they lateralize (negative over right sensorimotor, positive left). |
| 04 | `04_model_comparison.ipynb` | CSP+LDA vs bandpower+LDA vs SVM; LOSO accuracy + AUC. |

Final code must live in `src/bci_grasp/` — notebooks are for exploration
only. When a notebook converges on a stable analysis, port the logic into
the package and delete the notebook cells that duplicate it.
