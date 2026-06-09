# NumCompute-Stream

A modular, NumPy-only streaming machine learning framework built from scratch.  
**Author:** Saadman Sakib | **Course:** COMP 5004 | **Assignment:** 2.2

---

## Overview

NumCompute-Stream extends the NumCompute package (Assignment 2.1) into a full streaming ML framework. Every component supports incremental `partial_fit()` or `update()` methods, simulating real-world online learning where data arrives in chunks.

**New modules added:**
- `tree.py` — Decision Tree classifier (Gini / Entropy, depth-limited)
- `ensemble.py` — Ensemble classifier (Random Forest, Bagging)
- `stream.py` — StreamTrainer for chunk-wise training and logging
- `visualise.py` — matplotlib plotting for streaming metrics

**Updated modules:**
- `preprocessing.py` — `StandardScaler`, `OneHotEncoder`, `SimpleImputer` all support `partial_fit()`
- `stats.py` — `StreamingStats` with `update_stats()` API
- `metrics.py` — `StreamingMetrics` with `update()`, `reset()`, `result()`
- `pipeline.py` — `Pipeline.partial_fit()` for chained streaming

---

## Setup

### Requirements
- Python ≥ 3.8
- NumPy
- matplotlib
- pytest (for tests)
- jupyter (for demo notebook)

### Installation

```bash
# Clone the repository
git clone https://github.com/SAADMAN-S/numcompute-stream.git
cd numcompute-stream

# Install the package in editable mode
python -m pip install -e .

# Install dependencies
python -m pip install numpy matplotlib pytest jupyter
```

---

## Project Structure

```
numcompute-stream/
├── numcompute_stream/
│   ├── __init__.py
│   ├── io.py               # CSV loading with NaN handling
│   ├── preprocessing.py    # Scalers, Imputer, Encoder (+ partial_fit)
│   ├── stats.py            # Descriptive stats + StreamingStats
│   ├── metrics.py          # Classification metrics + StreamingMetrics
│   ├── pipeline.py         # Pipeline with partial_fit support
│   ├── tree.py             # DecisionTreeClassifier (NEW)
│   ├── ensemble.py         # EnsembleClassifier: Random Forest, Bagging (NEW)
│   ├── stream.py           # StreamTrainer for chunk-wise training (NEW)
│   ├── visualise.py        # matplotlib plotting functions (NEW)
│   ├── utils.py            # Distance functions, activations, helpers
│   ├── sort_search.py      # Sorting and search algorithms
│   ├── rank.py             # Ranking and percentiles
│   ├── optim.py            # Finite-difference optimisation
│   └── benchmarking.py     # Benchmarking harness
├── tests/
│   ├── test_tree.py        # DecisionTreeClassifier tests
│   ├── test_visualise.py   # Visualise tests
│   ├── test_ensemble.py    # EnsembleClassifier tests
│   ├── test_stream.py      # StreamTrainer tests
│   ├── test_streaming.py   # Streaming extensions tests
│   ├── test_metrics.py
│   ├── test_preprocessing.py
│   ├── test_stats.py
│   └── ...
├── demo/
│   ├── stream_demo.ipynb   # Streaming ML demo notebook
│   └── employees.csv       # Sample dataset
├── benchmark/
│   └── benchmark_streaming.py  # Performance benchmarks
├── README.md
└── pyproject.toml
```

---

## Running the Tests

```bash
python -m pytest tests/ -v
```

Expected output: **294 tests passing**

---

## Running the Benchmark

```bash
python benchmark/benchmark_streaming.py
```

Benchmarks include:
- Decision Tree vs Random Forest streaming fit time and accuracy
- Impact of chunk size on training time
- Python loop vs NumPy vectorised operations (81x speedup)
- Memory footprint growth over streaming chunks

---

## Running the Demo Notebook

```bash
cd demo
python -m jupyter notebook stream_demo.ipynb
```

The notebook demonstrates:
1. Loading CSV data via `io.py`
2. Splitting into chunks to simulate a data stream
3. Incremental training with `DecisionTreeClassifier` and `EnsembleClassifier`
4. Per-chunk accuracy logging via `StreamTrainer`
5. Real-time metric visualisation via `visualise.py`
6. `StreamingMetrics` for cumulative and rolling-window evaluation
7. `Pipeline.partial_fit()` chaining scaler and model

---

## Quick API Example

```python
import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.stream import StreamTrainer

# Create a streaming dataset
X = np.random.randn(200, 4)
y = (X[:, 0] > 0).astype(int)
chunks = [(X[i:i+50], y[i:i+50]) for i in range(0, 200, 50)]

# Train a Decision Tree incrementally
trainer = StreamTrainer(model=DecisionTreeClassifier(max_depth=5))
for Xc, yc in chunks:
    trainer.fit_chunk(Xc, yc)

trainer.print_log()

# Train a Random Forest incrementally
rf_trainer = StreamTrainer(
    model=EnsembleClassifier(n_estimators=10, method='random_forest')
)
for Xc, yc in chunks:
    rf_trainer.fit_chunk(Xc, yc)

print(rf_trainer.cumulative_metrics())
```

---

## Visualisation

```python
from numcompute_stream import visualise

# Plot accuracy over chunks
visualise.plot_metric_over_time(trainer.accuracy_history(), ylabel='Accuracy')

# Compare two models
visualise.compare_models(
    trainer.accuracy_history(),
    rf_trainer.accuracy_history(),
    labels=('Decision Tree', 'Random Forest')
)
```

---

## Constraints

- Only **NumPy** and **matplotlib** used for all ML and visualisation logic
- No scikit-learn, pandas, or PyTorch