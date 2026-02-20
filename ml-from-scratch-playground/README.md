# ml-from-scratch-playground

Learn machine learning fundamentals by implementing core algorithms **from scratch** using only:

- Python 3.10+
- NumPy
- Pandas
- Matplotlib

> ✅ No `sklearn` model implementations are used.

---

## 1) Project Goals

This repository is designed as a practical ML learning playground where you can:

- understand how popular ML algorithms work internally,
- train them on small synthetic datasets,
- inspect losses and evaluation metrics,
- visualize learning curves and decision boundaries.

---

## 2) Repository Layout

```text
ml-from-scratch-playground/
├── datasets/
│   ├── synthetic_regression.csv
│   ├── synthetic_classification.csv
│
├── ml_algorithms/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── gradient_descent.py
│
├── utils/
│   ├── metrics.py
│   ├── data_loader.py
│   ├── visualization.py
│
├── notebooks/
│   ├── 01_linear_regression_demo.ipynb
│   ├── 02_logistic_regression_demo.ipynb
│
├── examples/
│   ├── train_regression.py
│   ├── train_classification.py
│
├── requirements.txt
├── README.md
└── setup.py
```

Also included: `datasets/generate_datasets.py` to regenerate CSV datasets with a fixed random seed.

---

## 3) Implemented Algorithms

### Linear Regression (`ml_algorithms/linear_regression.py`)
- Gradient descent optimization
- Mean Squared Error (MSE) loss
- `fit()` / `predict()`

### Logistic Regression (`ml_algorithms/logistic_regression.py`)
- Sigmoid activation
- Binary cross-entropy loss
- `fit()` / `predict_proba()` / `predict()`

### k-Nearest Neighbors (`ml_algorithms/knn.py`)
- Euclidean distance
- Configurable `k`
- Majority voting

### Decision Tree - Basic Classifier (`ml_algorithms/decision_tree.py`)
- Gini impurity
- Binary splits only
- Max depth + minimum split size controls

### Gradient Descent Visualizer (`ml_algorithms/gradient_descent.py`)
- Utility to plot loss reduction across epochs

---

## 4) Utilities

### `utils/metrics.py`
- `mean_squared_error`
- `accuracy`
- `precision`
- `recall`

### `utils/data_loader.py`
- CSV loading helper
- `train_test_split` with reproducible random seed

### `utils/visualization.py`
- Scatter plots
- Loss curves
- 2D decision boundary plots

---

## 5) Synthetic Datasets

### Regression dataset (`datasets/synthetic_regression.csv`)
- 500 samples
- columns: `feature1, feature2, target`
- generated from:

\[
\text{target} = 3 \cdot \text{feature1} + 2 \cdot \text{feature2} + \text{noise}
\]

### Classification dataset (`datasets/synthetic_classification.csv`)
- 500 samples
- columns: `feature1, feature2, label`
- binary labels from two separable Gaussian clusters

To regenerate:

```bash
python datasets/generate_datasets.py
```

---

## 6) Local Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

---

## 7) Run Example Scripts

From repository root:

```bash
python examples/train_regression.py
python examples/train_classification.py
```

Expected outputs include training loss and metrics (MSE for regression, accuracy/precision/recall for classification).

---

## 8) Run Notebooks

```bash
jupyter notebook
```

Open:
- `notebooks/01_linear_regression_demo.ipynb`
- `notebooks/02_logistic_regression_demo.ipynb`

Notebook demos cover dataset loading, model training, and visualizations.

---

## 9) Screenshots Placeholder

Add generated plots/screenshots here:

- `![Linear Regression Loss](path/to/linear_loss.png)`
- `![Logistic Decision Boundary](path/to/logistic_boundary.png)`

---

## 10) Coding Standards

- PEP 8
- Type hints
- Docstrings
- Reproducible random seeds
