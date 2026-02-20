"""Logistic Regression implemented from scratch with gradient descent."""

from __future__ import annotations

import numpy as np


class LogisticRegression:
    """Binary Logistic Regression model."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Compute sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross entropy loss."""
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Train model using gradient descent."""
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_epochs):
            linear_output = x @ self.weights + self.bias
            y_pred = self._sigmoid(linear_output)

            loss = self._binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            dw = (1.0 / n_samples) * (x.T @ (y_pred - y))
            db = float((1.0 / n_samples) * np.sum(y_pred - y))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probability for positive class."""
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict_proba().")
        return self._sigmoid(x @ self.weights + self.bias)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        return (self.predict_proba(x) >= threshold).astype(int)
