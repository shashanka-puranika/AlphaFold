"""Visualization helpers for datasets and training curves."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str = "Feature 1",
    ylabel: str = "Feature 2",
) -> None:
    """Plot a scatter chart for 2D data."""
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()


def plot_loss_curve(loss_history: list[float], title: str = "Training Loss") -> None:
    """Plot model loss over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color="tab:green", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
) -> None:
    """Plot decision boundary for binary classifiers in 2D."""
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.8)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()
