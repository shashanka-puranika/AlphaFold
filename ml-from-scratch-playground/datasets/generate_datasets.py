"""Generate synthetic regression and classification datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_regression_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic linear regression data."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-10, 10, n_samples)
    x2 = rng.uniform(-5, 5, n_samples)
    noise = rng.normal(0, 2.0, n_samples)
    y = 3 * x1 + 2 * x2 + noise

    return pd.DataFrame({"feature1": x1, "feature2": x2, "target": y})


def generate_classification_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic binary classification data with two separable clusters."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2

    cluster_0 = rng.normal(loc=[-2, -2], scale=[1.0, 1.0], size=(half, 2))
    cluster_1 = rng.normal(loc=[2, 2], scale=[1.0, 1.0], size=(n_samples - half, 2))

    x = np.vstack([cluster_0, cluster_1])
    y = np.hstack([np.zeros(half, dtype=int), np.ones(n_samples - half, dtype=int)])

    return pd.DataFrame({"feature1": x[:, 0], "feature2": x[:, 1], "label": y})


def main() -> None:
    """Generate and persist datasets as CSV files."""
    dataset_dir = Path(__file__).resolve().parent

    regression_df = generate_regression_dataset()
    classification_df = generate_classification_dataset()

    regression_df.to_csv(dataset_dir / "synthetic_regression.csv", index=False)
    classification_df.to_csv(dataset_dir / "synthetic_classification.csv", index=False)

    print("Generated datasets in:", dataset_dir)


if __name__ == "__main__":
    main()
