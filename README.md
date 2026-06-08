# NumCompute

**A modular, NumPy-only machine-learning framework built from scratch.**

NumCompute is a self-contained Python package that re-implements the core scikit-learn workflow using only NumPy: CSV ingestion, preprocessing transformers, sorting and search primitives, ranking and percentiles, descriptive statistics, classification and regression metrics, finite-difference optimisation, and a chained Transformer/Estimator/Pipeline API.


---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Quickstart](#quickstart)
4. [API Overview](#api-overview)
5. [Performance](#performance)
6. [Testing](#testing)
7. [Demo Notebook](#demo-notebook)
8. [Authors](#authors)

---

## Installation

NumCompute targets Python 3.8+ and depends only on NumPy.

```bash
# Clone the repository
git clone https://github.com/BZMYSR/ProgAI-Assignment-2.1 NumCompute
cd NumCompute

# Create Virtual Environment:
python -m venv .venv

# Activate Environment:
# Windows: 
.venv\Scripts\activate
# macOS/Linux: 
source .venv/bin/activate
# Upon activation, (.venv) will appear in your terminal prompt.

# Install in editable mode
pip install -e .
pip install numpy
pip install matplotlib pytest  # Optional: for demo and testing
```

Or run the demo without installing — just make sure the package directory is on `PYTHONPATH`:

```bash
PYTHONPATH=. python demo/quickstart.py
```

**Requirements**

- Python ≥ 3.8
- NumPy ≥ 1.24
- (optional) Matplotlib ≥ 3.7 — only for the demo notebook
- (optional) pytest ≥ 7.0 — only for the test suite

---

## Project Structure

```
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py              # CSV reader (NaN fill, dtype handling)
│   ├── preprocessing.py   # StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
│   ├── sort_search.py     # stable_sort, multi_key_sort, topk, quickselect, binary_search
│   ├── rank.py            # Ranking with ties; percentiles
│   ├── stats.py           # Welford streaming, histogram, quantiles, summary
│   ├── metrics.py         # accuracy / precision / recall / F1, MSE, confusion matrix
│   ├── optim.py           # Finite-difference gradient (∇), Jacobian
│   ├── pipeline.py        # Transformer / Estimator / Pipeline API
│   ├── utils.py           # Distances, activations, logsumexp, batching
│   └── benchmarking.py    # Micro-benchmark harness
├── tests/                 # Unit tests covering edge cases
├── demo/
│   ├── quickstart.ipynb   # End-to-end demo (executed)
│   └── employees.csv      # Sample dataset with missing values
├── README.md
└── pyproject.toml
```

---

## API Overview

### `io.py` — Data ingestion

| Function | Purpose |
|---|---|
| `load_csv(path, delimiter=',', fill_missing_value='nan')` | Load CSV; cast to `float` if possible, fall back to string |
| `get_column(name, data, header)` | Extract a column by name |

### `preprocessing.py` — Feature preparation

| Class | Purpose |
|---|---|
| `SimpleImputer()` | Replace NaNs with constant value |
| `StandardScaler()` | Zero mean, unit variance per feature |
| `MinMaxScaler(feature_range=(0,1))` | Scale features to a target range |
| `OneHotEncoder()` | Expand categorical columns into 0/1 indicator columns |

All transformers expose `fit`, `transform`, and `fit_transform`.

### `sort_search.py` — Sorting & selection

| Function | Complexity | Purpose |
|---|---|---|
| `stable_sort(arr, axis=-1)` | O(n log n) | Stable mergesort |
| `multi_key_sort(arr, keys, ascending)` | O(n log n) | Sort 2-D array by multiple columns |
| `topk(arr, k, largest=True)` | O(n + k log k) | k-largest or k-smallest values |
| `quickselect(arr, k)` | O(n) average | k-th smallest (educational, pure Python) |
| `binary_search(arr, x)` | O(log n) | Returns `(index, found)` |

### `rank.py` — Ranking & percentiles

| Function | Purpose |
|---|---|
| `rank(data, method='average'\|'dense'\|'ordinal')` | Vectorised rank with tie handling |
| `percentile(data, q, interpolation='linear'\|'lower'\|'higher'\|'midpoint')` | NaN-safe percentile |

### `stats.py` — Descriptive & streaming statistics

| Function / Class | Purpose |
|---|---|
| `mean / var / std / median / minimum / maximum` | NaN-safe (`skipna=True` default) |
| `histogram(x, bins=10)` | NaN-safe histogram |
| `quantile(x, q, interpolation='linear')` | NaN-safe quantile |
| `summary(x, axis=None)` | Returns dict of common stats |
| `Welford()` | Single-pass online mean / variance |

### `metrics.py` — Evaluation metrics

| Function | Purpose |
|---|---|
| `accuracy(y_true, y_pred)` | Classification accuracy |
| `precision / recall / f1(..., average='binary'\|'macro'\|None)` | Standard classification metrics |
| `confusion_matrix(y_true, y_pred, labels=None)` | Square matrix of class counts |
| `mse(y_true, y_pred)` | Regression error metrics |

### `optim.py` — Finite-difference optimisation

| Function | Purpose |
|---|---|
| `grad(f, x, h=1e-5, method='central'\|'forward')` | Gradient of a scalar function |
| `jacobian(F, x, h=1e-5, method='central'\|'forward')` | Jacobian of a vector function |

Central-difference is **O(h²)** accurate; forward-difference is **O(h)** accurate.

### `pipeline.py` — Composition

| Class | Purpose |
|---|---|
| `Transformer` | Base class with `fit / transform / fit_transform` |
| `Estimator` | Base class with `fit / predict` |
| `Pipeline([(name, step), ...])` | Chains transformers and an optional final estimator |

The Pipeline uses `inspect.signature` to detect whether a step accepts `y`, so unsupervised transformers and supervised estimators can be chained without special-casing.

### `utils.py` — General helpers

Distances (`euclidean`, `manhattan`, `cosine_distance`), activations (`relu`, `leaky_relu`, `sigmoid`, `tanh`), `softmax`, `logsumexp` (max-shifted for numerical stability), `batch` generator.

---

## Performance

All benchmarks run single-threaded via `time.perf_counter` (n_repeat = 7, warm-up = 1); medians are reported. Environment: Python 3.12, NumPy 2.4, Linux.

### Vectorised vs Python loops (n = 100,000)

| Operation | Python loop | Vectorised | Speedup |
|---|---:|---:|---:|
| `mean` | 85.81 ms | 524 µs | **163.7×** |
| `accuracy` | 187.58 ms | 2.29 ms | **81.8×** |
| sum-of-squared-diff | 193.39 ms | 23.90 ms | **8.1×** |

### Sorting & search

| Operation | n | Median | Notes |
|---|---:|---:|---|
| `stable_sort` | 100,000 | 117.92 ms | NumPy mergesort, O(n log n) |
| `np.sort` (reference) | 100,000 | 10.18 ms | Quicksort baseline |
| `topk` (k = 10) | 100,000 | 3.62 ms | argpartition — O(n) |
| `topk` (k = 10,000) | 100,000 | 4.93 ms | argpartition + argsort(k) |
| `binary_search` | 100,000 | 2.11 µs | searchsorted — O(log n) |

### Gradient accuracy

For *f(x) = ‖x‖²* with x ∈ ℝ⁵⁰ and h = 10⁻⁵:

| Method | Max abs error | Theory |
|---|---:|---|
| Forward difference | 1.00 × 10⁻⁵ | O(h) |
| Central difference | **4.27 × 10⁻¹⁰** | O(h²) |

A linear map *F(x) = Ax* with A ∈ ℝ²⁰ˣ⁵⁰ is recovered to within **1.6 × 10⁻¹⁰** entry-wise.

Reproduce these numbers with:

```bash
python -m numcompute.benchmarking
```

---

## Testing

Run the full suite with pytest:

```bash
pytest tests/ -v
```

---

## Demo Notebook

The demo (`demo/quickstart.ipynb`) walks through the full pipeline on a fictional 60-employee dataset with realistic missing values:

1. Loading a CSV with `io.load_csv`
2. Imputation → standardisation → one-hot encoding
3. Top-k highest salaries, quickselect median, binary search
4. Tie-aware ranking and salary percentiles
5. Per-column statistics, histogram, Welford streaming
6. Finite-difference gradient and Jacobian
7. Vectorised vs. Python-loop benchmark
8. Pipeline composition

The notebook is pre-executed — open it in JupyterLab or VS Code to see the results

---

## Authors

| Author | Modules Owned |
|---|---|
| Risat Rahaman | `io.py`, `preprocessing.py`, `utils.py` |
| Saadman Sakib | `sort_search.py`, `rank.py` |
| Mazharul Islam Rakib | `optim.py`, `pipeline.py` |
| Benzamin Yasir | `stats.py`, `metrics.py`|
| Team | `benchmarking.py`, `tests/` |

---