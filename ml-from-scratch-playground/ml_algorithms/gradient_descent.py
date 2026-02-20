"""Utilities for gradient descent visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_loss_over_epochs(loss_history: list[float], title: str = "Loss over Epochs") -> None:
    """Plot gradient descent loss reduction across epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color="tab:blue", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
