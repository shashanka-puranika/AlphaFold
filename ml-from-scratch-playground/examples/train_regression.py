"""Train linear regression model on synthetic regression dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_algorithms.linear_regression import LinearRegression
from utils.data_loader import load_csv, train_test_split
from utils.metrics import mean_squared_error


def main() -> None:
    """Run regression training example."""
    np.random.seed(42)
    data_path = PROJECT_ROOT / "datasets" / "synthetic_regression.csv"
    df = load_csv(data_path)

    x = df[["feature1", "feature2"]].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_seed=42)

    model = LinearRegression(learning_rate=0.005, n_epochs=2000)
    model.fit(x_train, y_train)

    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    train_loss = mean_squared_error(y_train, train_preds)
    test_loss = mean_squared_error(y_test, test_preds)

    print(f"Final training loss (MSE): {train_loss:.4f}")
    print(f"Test MSE: {test_loss:.4f}")


if __name__ == "__main__":
    main()
