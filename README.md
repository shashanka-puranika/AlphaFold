# AlphaFold

Bioinformatics and machine-learning playground repository.

## Overview

This repository currently contains:

- `ml-from-scratch-playground/`: a complete educational project for learning ML fundamentals by implementing algorithms from scratch using Python, NumPy, Pandas, and Matplotlib.
- `AlphaFold2.ipynb`: legacy notebook asset retained from earlier work.

## Main Project: `ml-from-scratch-playground`

Inside `ml-from-scratch-playground/` you will find:

- from-scratch implementations of:
  - Linear Regression
  - Logistic Regression
  - k-Nearest Neighbors (kNN)
  - Decision Tree (basic classifier)
- synthetic regression/classification datasets
- utility modules for metrics, data loading, train/test split, and visualization
- runnable training scripts in `examples/`
- demo notebooks in `notebooks/`

## Quick Start

```bash
cd ml-from-scratch-playground
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/train_regression.py
python examples/train_classification.py
```

## Repository Structure (top level)

```text
.
├── AlphaFold2.ipynb
├── README.md
└── ml-from-scratch-playground/
```

## Notes

- Python 3.10+ is recommended.
- The ML project intentionally avoids scikit-learn model implementations for educational clarity.
