from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np

from .config import CHECKPOINTS_DIR, FIGURES_DIR, RESULTS_DIR


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    for directory in (RESULTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def clean_generated_outputs() -> None:
    ensure_directories()

    for directory, patterns in (
        (RESULTS_DIR, ("*.csv", "*.txt", "*.png")),
        (CHECKPOINTS_DIR, ("*.pkl",)),
        (FIGURES_DIR, ("*.png",)),
    ):
        for pattern in patterns:
            for path in directory.glob(pattern):
                path.unlink()


def remove_notebook_checkpoints(root: Path) -> None:
    for checkpoint_dir in root.rglob(".ipynb_checkpoints"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root.")
