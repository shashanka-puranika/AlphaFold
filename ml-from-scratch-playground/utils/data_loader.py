"""Data loading and splitting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_seed: int = 42,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train and test sets."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    n_samples = x.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)

    split_idx = int(n_samples * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]
