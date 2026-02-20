"""Evaluation metrics implemented from scratch."""

from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary precision."""
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positive = np.sum(y_pred == 1)
    return float(true_positive / predicted_positive) if predicted_positive > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary recall."""
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    actual_positive = np.sum(y_true == 1)
    return float(true_positive / actual_positive) if actual_positive > 0 else 0.0
