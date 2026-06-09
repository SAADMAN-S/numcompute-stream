"""
benchmark_streaming.py

Performance benchmarks for NumCompute-Stream.

Compares:
1. Single DecisionTree vs EnsembleClassifier (Random Forest) under streaming
2. Loop-based prediction vs vectorised NumPy operations

Run with: python benchmark/benchmark_streaming.py

Author: Saadman Sakib
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.preprocessing import StandardScaler

# Helpers

def make_dataset(n_samples=400, n_features=10, n_classes=2, seed=42):
    """Generate a reproducible synthetic classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    w = rng.standard_normal(n_features)
    logits = X @ w
    thresholds = np.linspace(logits.min(), logits.max(), n_classes + 1)
    y = np.digitize(logits, thresholds[1:-1])
    return X, y


def time_it(fn, n_repeat=2):
    """Return median wall-clock time over n_repeat calls (seconds)."""
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def print_header(title):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_row(label, value, unit="s", width=45):
    print(f"  {label:<{width}} {value:.6f} {unit}")


# Benchmark 1: Single Tree vs Random Forest — streaming fit

def bench_tree_vs_forest():
    print_header("Benchmark 1: Decision Tree vs Random Forest (streaming fit)")

    X, y = make_dataset(n_samples=400, n_features=10)
    chunk_size = 200
    chunks = [(X[i:i+chunk_size], y[i:i+chunk_size])
              for i in range(0, len(X), chunk_size)]

    tree = DecisionTreeClassifier(max_depth=5)
    forest = EnsembleClassifier(n_estimators=5, method="random_forest",
                                max_depth=5, random_state=0)

    def stream_tree():
        t = DecisionTreeClassifier(max_depth=5)
        for Xc, yc in chunks:
            t.partial_fit(Xc, yc)

    def stream_forest():
        f = EnsembleClassifier(n_estimators=5, method="random_forest",
                               max_depth=5, random_state=0)
        for Xc, yc in chunks:
            f.partial_fit(Xc, yc)

    t_tree = time_it(stream_tree)
    t_forest = time_it(stream_forest)

    # Fit once for accuracy comparison
    for Xc, yc in chunks:
        tree.partial_fit(Xc, yc)
        forest.partial_fit(Xc, yc)

    acc_tree = float(np.mean(tree.predict(X) == y))
    acc_forest = float(np.mean(forest.predict(X) == y))

    print(f"\n  {'Model':<30} {'Fit time (s)':>15} {'Accuracy':>12}")
    print(f"  {'-'*58}")
    print(f"  {'DecisionTree (depth=5)':<30} {t_tree:>15.4f} {acc_tree:>12.4f}")
    print(f"  {'RandomForest (10 trees)':<30} {t_forest:>15.4f} {acc_forest:>12.4f}")
    print(f"\n  Speedup factor (tree / forest): {t_tree/t_forest:.2f}x")



# Benchmark 2: Streaming chunk size impact

def bench_chunk_size():
    print_header("Benchmark 2: Impact of chunk size on streaming fit time")

    X, y = make_dataset(n_samples=400, n_features=10)
    chunk_sizes = [50, 100, 200, 500, 1000]

    print(f"\n  {'Chunk size':<15} {'# Chunks':<12} {'Fit time (s)':>12}")
    print(f"  {'-'*42}")

    for cs in chunk_sizes:
        chunks = [(X[i:i+cs], y[i:i+cs]) for i in range(0, len(X), cs)]

        def run():
            clf = DecisionTreeClassifier(max_depth=4)
            for Xc, yc in chunks:
                clf.partial_fit(Xc, yc)

        t = time_it(run)
        print(f"  {cs:<15} {len(chunks):<12} {t:>12.4f}")


# Benchmark 3: Loop-based vs vectorised prediction

def bench_loop_vs_vectorised():
    print_header("Benchmark 3: Loop-based vs vectorised accuracy computation")

    n = 100_000
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)

    def loop_accuracy():
        correct = 0
        for t, p in zip(y_true, y_pred):
            if t == p:
                correct += 1
        return correct / n

    def vec_accuracy():
        return float(np.mean(y_true == y_pred))

    t_loop = time_it(loop_accuracy, n_repeat=3)
    t_vec = time_it(vec_accuracy, n_repeat=3)

    print(f"\n  {'Method':<35} {'Time (s)':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Python loop (n=100,000)':<35} {t_loop:>12.6f}")
    print(f"  {'NumPy vectorised (n=100,000)':<35} {t_vec:>12.6f}")
    print(f"\n  Speedup: {t_loop/t_vec:.1f}x faster with vectorisation")


# Benchmark 4: StreamTrainer memory footprint over time

def bench_memory_footprint():
    print_header("Benchmark 4: Model memory footprint as chunks accumulate")

    X, y = make_dataset(n_samples=400, n_features=10)
    chunk_size = 200
    chunks = [(X[i:i+chunk_size], y[i:i+chunk_size])
              for i in range(0, len(X), chunk_size)]

    trainer = StreamTrainer(model=DecisionTreeClassifier(max_depth=4))

    print(f"\n  {'Chunk':<8} {'Samples seen':>14} {'Memory (bytes)':>16} {'Accuracy':>10}")
    print(f"  {'-'*52}")

    for Xc, yc in chunks:
        trainer.fit_chunk(Xc, yc)
        entry = trainer.log_[-1]
        print(f"  {entry['chunk']:<8} {entry['chunk'] * chunk_size:>14} "
              f"{entry['memory_bytes']:>16} {entry['accuracy']:>10.4f}")


# Main

if __name__ == "__main__":
    print("\nNumCompute-Stream — Streaming Benchmarks")
    print("Python version: " + str(sys.version.split()[0]))

    import numpy as np
    print("NumPy version: " + np.__version__)

    bench_tree_vs_forest()
    bench_chunk_size()
    bench_loop_vs_vectorised()
    bench_memory_footprint()

    print("\n" + "=" * 70)
    print("  Benchmarks complete.")
    print("=" * 70 + "\n")