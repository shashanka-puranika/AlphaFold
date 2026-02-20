"""Train logistic regression model on synthetic classification dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_algorithms.logistic_regression import LogisticRegression
from utils.data_loader import load_csv, train_test_split
from utils.metrics import accuracy, precision, recall


def main() -> None:
    """Run classification training example."""
    np.random.seed(42)
    data_path = PROJECT_ROOT / "datasets" / "synthetic_classification.csv"
    df = load_csv(data_path)

    x = df[["feature1", "feature2"]].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_seed=42)

    model = LogisticRegression(learning_rate=0.1, n_epochs=1000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    final_loss = model.loss_history[-1]

    print(f"Final training loss (BCE): {final_loss:.4f}")
    print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
    print(f"Precision: {precision(y_test, y_pred):.4f}")
    print(f"Recall: {recall(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
