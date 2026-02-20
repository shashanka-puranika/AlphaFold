"""Linear Regression implemented from scratch with gradient descent."""

from __future__ import annotations

import numpy as np


class LinearRegression:
    """Linear Regression model trained via gradient descent."""

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Train the model using gradient descent.

        Args:
            x: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            The trained model instance.
        """
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_epochs):
            y_pred = x @ self.weights + self.bias
            errors = y_pred - y

            loss = float(np.mean(errors**2))
            self.loss_history.append(loss)

            dw = (2.0 / n_samples) * (x.T @ errors)
            db = float((2.0 / n_samples) * np.sum(errors))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict target values for input features."""
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
        return x @ self.weights + self.bias
