"""Basic binary Decision Tree classifier implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """A node in the decision tree."""

    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    value: int | None = None


class DecisionTreeClassifier:
    """Decision Tree classifier using Gini impurity and binary splits."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Node | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Build tree from training data."""
        self.root = self._build_tree(x, y, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        if self.root is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
        return np.array([self._traverse_tree(sample, self.root) for sample in x])

    def _gini(self, y: np.ndarray) -> float:
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return float(1.0 - np.sum(probabilities**2))

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        n_samples, n_features = x.shape
        if n_samples < self.min_samples_split:
            return None, None

        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature_idx in range(n_features):
            thresholds = np.unique(x[:, feature_idx])
            for threshold in thresholds:
                left_mask = x[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (
                    (left_mask.sum() / n_samples) * gini_left
                    + (right_mask.sum() / n_samples) * gini_right
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> Node:
        if len(np.unique(y)) == 1:
            return Node(value=int(y[0]))

        if depth >= self.max_depth or len(y) < self.min_samples_split:
            majority_class = int(np.bincount(y).argmax())
            return Node(value=majority_class)

        feature_idx, threshold = self._best_split(x, y)
        if feature_idx is None or threshold is None:
            majority_class = int(np.bincount(y).argmax())
            return Node(value=majority_class)

        left_mask = x[:, feature_idx] <= threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(x[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_index=feature_idx,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _traverse_tree(self, sample: np.ndarray, node: Node) -> int:
        if node.value is not None:
            return node.value

        if sample[node.feature_index] <= node.threshold:
            return self._traverse_tree(sample, node.left)
        return self._traverse_tree(sample, node.right)
