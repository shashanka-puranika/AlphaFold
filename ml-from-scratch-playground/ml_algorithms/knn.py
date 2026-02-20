"""k-Nearest Neighbors classifier implemented from scratch."""

from __future__ import annotations

import numpy as np


class KNNClassifier:
    """Simple kNN classifier using Euclidean distance."""

    def __init__(self, k: int = 3) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store the training dataset."""
        self.x_train = x
        self.y_train = y
        return self

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return float(np.sqrt(np.sum((x1 - x2) ** 2)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples."""
        if self.x_train is None or self.y_train is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")

        predictions = [self._predict_single(sample) for sample in x]
        return np.array(predictions)

    def _predict_single(self, sample: np.ndarray) -> int:
        distances = [self._euclidean_distance(sample, x_train_i) for x_train_i in self.x_train]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_indices]
        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return int(labels[np.argmax(counts)])
