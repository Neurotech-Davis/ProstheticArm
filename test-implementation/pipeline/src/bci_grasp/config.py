"""Config loading.

Single entry point for reading the YAML files in ``configs/``. Scripts and
modules should load their config once at the top of ``main()`` and pass the
resulting dict down — no module-level magic.

Example
-------
    from bci_grasp.config import load_config
    cfg = load_config("data")                 # loads configs/data.yaml
    cfg_all = load_config()                   # loads every YAML in configs/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# configs/ lives at <pipeline_root>/configs; this file is at
# <pipeline_root>/src/bci_grasp/config.py, so parents[2] == pipeline_root.
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PIPELINE_ROOT / "configs"


def load_config(name: str | None = None) -> dict[str, Any]:
    """Load a named YAML config, or all configs merged into one dict.

    Parameters
    ----------
    name : str or None
        Base filename without extension (e.g. ``"data"`` for ``data.yaml``).
        If ``None``, every ``*.yaml`` in ``configs/`` is loaded and returned
        under its filename stem.

    Returns
    -------
    dict
        The parsed YAML content. When ``name`` is None, the top-level keys
        are the YAML filename stems (``data``, ``preprocessing``, ...).
    """
    if name is None:
        return {
            path.stem: yaml.safe_load(path.read_text())
            for path in sorted(CONFIG_DIR.glob("*.yaml"))
        }

    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


def resolve_repo_path(relative: str | Path) -> Path:
    """Resolve a repo-root-relative path (as stored in configs) to an absolute Path.

    Configs store paths like ``test-implementation/ds003810`` relative to the
    ProstheticArm repo root. This helper turns that into an absolute Path
    suitable for passing to mne-bids, etc.

    The repo root is the parent of ``test-implementation/``, which is the
    parent of ``pipeline/``. So repo_root = PIPELINE_ROOT.parents[1].
    """
    repo_root = PIPELINE_ROOT.parents[1]
    return (repo_root / Path(relative)).resolve()
