# ml-from-scratch-playground

A complete educational repository for learning machine learning fundamentals by implementing classic algorithms **from scratch** with only:

- Python 3.10+
- NumPy
- Pandas
- Matplotlib

> No scikit-learn model implementations are used.

---

## Project Overview

This project contains clean, well-documented implementations of core ML algorithms and utilities:

- **Linear Regression** (gradient descent + MSE)
- **Logistic Regression** (sigmoid + binary cross-entropy)
- **k-Nearest Neighbors (kNN)** (Euclidean distance)
- **Decision Tree (basic classifier)** (Gini impurity + binary splits)
- **Gradient Descent loss visualizer**

It also includes:

- synthetic datasets
- notebook demos
- training scripts
- plotting helpers
- train/test split and evaluation metrics

---

## Repository Structure

```text
ml-from-scratch-playground/
├── datasets/
│   ├── synthetic_regression.csv
│   ├── synthetic_classification.csv
│   └── generate_datasets.py
├── ml_algorithms/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── gradient_descent.py
├── utils/
│   ├── metrics.py
│   ├── data_loader.py
│   ├── visualization.py
├── notebooks/
│   ├── 01_linear_regression_demo.ipynb
│   ├── 02_logistic_regression_demo.ipynb
├── examples/
│   ├── train_regression.py
│   ├── train_classification.py
├── requirements.txt
├── README.md
└── setup.py
```

---

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional package install:

```bash
pip install -e .
```

---

## Generate Datasets

Synthetic datasets are generated with NumPy using a fixed random seed for reproducibility.

```bash
python datasets/generate_datasets.py
```

### Dataset details

- `synthetic_regression.csv`
  - columns: `feature1, feature2, target`
  - 500 samples
  - relation: `target = 3*feature1 + 2*feature2 + noise`

- `synthetic_classification.csv`
  - columns: `feature1, feature2, label`
  - 500 samples
  - binary labels from two separable Gaussian clusters

---

## Run Examples

From repository root:

```bash
python examples/train_regression.py
python examples/train_classification.py
```

Both scripts print final training loss and evaluation metrics.

---

## Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Open:

- `notebooks/01_linear_regression_demo.ipynb`
- `notebooks/02_logistic_regression_demo.ipynb`

Each notebook demonstrates:

- loading CSV datasets
- training from-scratch models
- plotting loss curves
- plotting classification decision boundaries

---

## Algorithm Explanations

### 1) Linear Regression

Minimizes Mean Squared Error:

\[
\mathcal{L} = \frac{1}{n}\sum_i(\hat{y}_i - y_i)^2
\]

Weights are optimized with gradient descent updates each epoch.

### 2) Logistic Regression

Uses sigmoid to map linear output to probability:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Optimized using binary cross-entropy loss.

### 3) kNN

A non-parametric method that predicts by majority vote of the `k` nearest training points under Euclidean distance.

### 4) Decision Tree (Basic)

Finds binary splits by minimizing weighted **Gini impurity**:

\[
Gini = 1 - \sum_c p(c)^2
\]

---

## Screenshots

_Add screenshots of notebook outputs / plots here._

- `![Linear Regression Loss](path/to/linear_loss.png)`
- `![Logistic Decision Boundary](path/to/logistic_boundary.png)`

---

## Coding Standards

- PEP 8
- Type hints
- Docstrings
- Reproducible random seeds

