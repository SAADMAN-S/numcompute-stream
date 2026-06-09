"""
benchmark.py - Micro-benchmark harness for NumCompute

"""
from __future__ import annotations

import statistics
import sys
import time
from typing import Any, Callable, Dict, List

import numpy as np

from numcompute.utils import (
    euclidean,
    manhattan,
    cosine_distance,
    relu,
    leaky_relu,
    sigmoid,
    tanh,
    softmax,
    logsumexp,
)



# Timing helpers

def format_time(seconds: float) -> str:
    """Format a duration in the most readable unit (s / ms / us / ns)."""
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.3f} us"
    return f"{seconds * 1e9:.3f} ns"


def time_function(
    func: Callable[..., Any],
    *args: Any,
    n_repeat: int = 7,
    n_inner: int = 1,
    warmup: int = 1,
    **kwargs: Any,
) -> Dict[str, float]:
    """Time 'func' and return summary statistics.

    Parameters:-
    func : callable
        Function to benchmark.
    *args, **kwargs :
        Arguments forwarded to `func`.
    n_repeat : int, default 7
        Number of independent timing measurements.
    n_inner : int, default 1
        How many calls to make per measurement (to increase timing resolution).
    warmup : int, default 1
        Untimed calls performed first (caches, JIT, allocator warm-up).

    Returns:-
    dict
        Keys: stats (all in seconds per single call), 
        plus n_repeat and n_inner for reference.

    """
    if n_repeat < 1:
        raise ValueError("n_repeat must be >= 1.")
    if n_inner < 1:
        raise ValueError("n_inner must be >= 1.")

    # Warm-up (results discarded)
    for _ in range(warmup):
        func(*args, **kwargs)

    times: List[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for _ in range(n_inner):
            func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) / n_inner)

    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "median": statistics.median(times),
        "n_repeat": n_repeat,
        "n_inner": n_inner,
    }



# Benchmark class
class Benchmark:
    """Collect, compare and pretty print microbenchmark results."""

    def __init__(self, name: str = "Benchmark") -> None:
        self.name = name
        self.results: List[Dict[str, Any]] = []

    def run(
        self,
        label: str,
        func: Callable[..., Any],
        *args: Any,
        n_repeat: int = 7,
        n_inner: int = 1,
        warmup: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Time a single function call and store the result under 'label'."""
        result = time_function(
            func, *args,
            n_repeat=n_repeat, n_inner=n_inner, warmup=warmup,
            **kwargs,
        )
        result["label"] = label
        self.results.append(result)
        return result

    def compare(self, faster: str, slower: str) -> float:
        a = next(r for r in self.results if r["label"] == faster)
        b = next(r for r in self.results if r["label"] == slower)
        return b["mean"] / a["mean"]

    def report(self) -> str:
        """Return a formatted table of all stored results"""
        if not self.results:
            return f"{self.name}: no results.\n"

        label_w = max(len(r["label"]) for r in self.results)
        label_w = max(label_w, len("Function"))
        total_w = label_w + 60

        header = (
            f"{'Function':<{label_w}}  "
            f"{'Mean':>12}  {'Std':>12}  {'Min':>12}  {'Repeats':>8}"
        )
        lines = [
            "",
            self.name,
            "=" * total_w,
            header,
            "-" * total_w,
        ]
        for r in self.results:
            lines.append(
                f"{r['label']:<{label_w}}  "
                f"{format_time(r['mean']):>12}  "
                f"{format_time(r['std']):>12}  "
                f"{format_time(r['min']):>12}  "
                f"{r['n_repeat']:>8}"
            )
        return "\n".join(lines) + "\n"

    def print_report(self) -> None:
        print(self.report())



# Shared loop implementations — mean, accuracy, ssd
def _mean_loop(arr: np.ndarray) -> float:
    total = 0.0
    for v in arr:
        total += float(v)
    return total / len(arr)


def _mean_vec(arr: np.ndarray) -> float:
    return float(np.mean(arr))


def _accuracy_loop(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = 0
    n = 0
    for a, b in zip(y_true, y_pred):
        n += 1
        if a == b:
            correct += 1
    return correct / n


def _accuracy_vec(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _ssd_loop(a: np.ndarray, b: np.ndarray) -> float:
    total = 0.0
    for x_i, y_i in zip(a, b):
        d = float(x_i) - float(y_i)
        total += d * d
    return total


def _ssd_vec(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sum(diff * diff))



# sort_search & rank loop implementations

def _stable_sort_loop(arr: np.ndarray) -> list:
    """Bubble sort — O(n2) stable sort, pure Python."""
    data = list(arr)
    n = len(data)
    for i in range(n):
        for j in range(n - i - 1):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data


def _stable_sort_vec(arr: np.ndarray) -> np.ndarray:
    """Vectorised stable sort using np.sort(kind='stable')."""
    return np.sort(arr, kind="stable")


def _topk_loop(arr: np.ndarray, k: int) -> list:
    """Pure Python top-k via full list sort — O(n log n) in Python."""
    data = list(arr)
    data.sort(reverse=True)
    return data[:k]


def _topk_vec(arr: np.ndarray, k: int) -> np.ndarray:
    """Vectorised top-k using np.argpartition — O(n) average."""
    idx = np.argpartition(arr, -k)[-k:]
    return arr[idx[np.argsort(arr[idx])[::-1]]]


def _linear_search_loop(arr: np.ndarray, x: float):
    """Pure Python linear search — O(n)."""
    for i, val in enumerate(arr):
        if val == x:
            return i, True
    return len(arr), False


def _binary_search_vec(arr: np.ndarray, x: float):
    """Vectorised binary search using np.searchsorted — O(log n)."""
    idx = int(np.searchsorted(arr, x, side="left"))
    found = idx < len(arr) and bool(arr[idx] == x)
    return idx, found


def _rank_loop(arr: np.ndarray) -> list:
    """Pure Python ordinal rank — O(n2)."""
    n = len(arr)
    ranks = [0] * n
    for i in range(n):
        r = 1
        for j in range(n):
            if arr[j] < arr[i]:
                r += 1
            elif arr[j] == arr[i] and j < i:
                r += 1
        ranks[i] = r
    return ranks


def _rank_vec(arr: np.ndarray) -> np.ndarray:
    """Vectorised ordinal rank using np.argsort — O(n log n)."""
    n = len(arr)
    sorter = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[sorter] = np.arange(1, n + 1, dtype=float)
    return ranks


def _percentile_loop(arr: np.ndarray, q: float) -> float:
    """Pure Python linear interpolation percentile."""
    data = sorted(float(x) for x in arr if x == x)
    n = len(data)
    virtual_idx = (q / 100.0) * (n - 1)
    lo = int(virtual_idx)
    hi = min(lo + 1, n - 1)
    return data[lo] + (virtual_idx - lo) * (data[hi] - data[lo])


def _percentile_vec(arr: np.ndarray, q: float) -> float:
    """Vectorised percentile using np.sort + index arithmetic."""
    data = arr[~np.isnan(arr)]
    n = len(data)
    s = np.sort(data, kind="stable")
    vi = (q / 100.0) * (n - 1)
    lo = int(np.floor(vi))
    hi = min(lo + 1, n - 1)
    return float(s[lo] + (vi - lo) * (s[hi] - s[lo]))



# utils.py loop implementations

def _euclidean_loop(a: np.ndarray, b: np.ndarray) -> float:
    total = 0.0
    for x_i, y_i in zip(a, b):
        d = float(x_i) - float(y_i)
        total += d * d
    return np.sqrt(total)


def _manhattan_loop(a: np.ndarray, b: np.ndarray) -> float:
    total = 0.0
    for x_i, y_i in zip(a, b):
        total += abs(float(x_i) - float(y_i))
    return total


def _cosine_loop(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x_i, y_i in zip(a, b):
        dot += float(x_i) * float(y_i)
        norm_a += float(x_i) * float(x_i)
        norm_b += float(y_i) * float(y_i)
    norm_a = max(np.sqrt(norm_a), eps)
    norm_b = max(np.sqrt(norm_b), eps)
    return 1.0 - dot / (norm_a * norm_b)


def _relu_loop(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    for i, x in enumerate(arr):
        out[i] = max(0.0, float(x))
    return out


def _leaky_relu_loop(arr: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    out = np.empty_like(arr)
    for i, x in enumerate(arr):
        v = float(x)
        out[i] = v if v > 0 else alpha * v
    return out


def _sigmoid_loop(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    for i, x in enumerate(arr):
        v = float(x)
        out[i] = 1.0 / (1.0 + np.exp(-v))
    return out


def _tanh_loop(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    for i, x in enumerate(arr):
        out[i] = np.tanh(float(x))
    return out


def _softmax_loop(arr: np.ndarray) -> np.ndarray:
    max_val = float(arr[0])
    for x in arr:
        if float(x) > max_val:
            max_val = float(x)
    exps = np.empty_like(arr)
    for i, x in enumerate(arr):
        exps[i] = np.exp(float(x) - max_val)
    s = np.sum(exps)
    for i in range(len(exps)):
        exps[i] /= s
    return exps


def _logsumexp_loop(arr: np.ndarray) -> float:
    max_val = float(arr[0])
    for x in arr:
        if float(x) > max_val:
            max_val = float(x)
    total = 0.0
    for x in arr:
        total += np.exp(float(x) - max_val)
    return max_val + np.log(total)



# Speedup pair lists


_LOOP_VS_VEC_PAIRS: List[tuple] = [
    ("mean",        "mean (Python loop)",        "mean (vectorised)"),
    ("accuracy",    "accuracy (Python loop)",    "accuracy (vectorised)"),
    ("sum-sq-diff", "sum-sq-diff (Python loop)", "sum-sq-diff (vectorised)"),
]

_SORT_RANK_PAIRS: List[tuple] = [
    ("stable_sort",   "stable_sort (Python loop)",   "stable_sort (vectorised)"),
    ("topk",          "topk (Python loop)",           "topk (vectorised)"),
    ("binary_search", "binary_search (Python loop)", "binary_search (vectorised)"),
    ("rank",          "rank (Python loop)",           "rank (vectorised)"),
    ("percentile",    "percentile (Python loop)",     "percentile (vectorised)"),
]

_UTILS_PAIRS: List[tuple] = [
    ("euclidean",  "euclidean (Python loop)",  "euclidean (vectorised)"),
    ("manhattan",  "manhattan (Python loop)",  "manhattan (vectorised)"),
    ("cosine",     "cosine (Python loop)",     "cosine (vectorised)"),
    ("relu",       "relu (Python loop)",       "relu (vectorised)"),
    ("leaky_relu", "leaky_relu (Python loop)", "leaky_relu (vectorised)"),
    ("sigmoid",    "sigmoid (Python loop)",    "sigmoid (vectorised)"),
    ("tanh",       "tanh (Python loop)",       "tanh (vectorised)"),
    ("softmax",    "softmax (Python loop)",    "softmax (vectorised)"),
    ("logsumexp",  "logsumexp (Python loop)",  "logsumexp (vectorised)"),
]



# Benchmark functions

def bench_loops_vs_vectorised(
    n_samples: int = 100_000,
    seed: int = 42,
) -> Benchmark:

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples)
    y = rng.standard_normal(n_samples)
    y_true = rng.integers(0, 2, size=n_samples)
    y_pred = rng.integers(0, 2, size=n_samples)

    assert np.isclose(_mean_loop(x), _mean_vec(x)), \
        "mean implementations disagree"
    assert _accuracy_loop(y_true, y_pred) == _accuracy_vec(y_true, y_pred), \
        "accuracy implementations disagree"
    assert np.isclose(_ssd_loop(x, y), _ssd_vec(x, y)), \
        "ssd implementations disagree"

    bm = Benchmark(f"vectorised vs Python loops (n={n_samples})")

    bm.run("mean (Python loop)",        _mean_loop,     x,
           n_repeat=3, warmup=0)
    bm.run("mean (vectorised)",         _mean_vec,      x)

    bm.run("accuracy (Python loop)",    _accuracy_loop, y_true, y_pred,
           n_repeat=3, warmup=0)
    bm.run("accuracy (vectorised)",     _accuracy_vec,  y_true, y_pred)

    bm.run("sum-sq-diff (Python loop)", _ssd_loop,      x, y,
           n_repeat=3, warmup=0)
    bm.run("sum-sq-diff (vectorised)",  _ssd_vec,       x, y)

    return bm


def bench_sort_search_rank(
    n_large: int = 100_000,
    n_small: int = 2_000,
    k: int = 10,
    seed: int = 42,
) -> Benchmark:
    """
    Benchmark sort_search and rank functions:
    vectorised NumPy vs pure Python loop equivalents.

    n_large : array size for topk, binary_search, percentile
    n_small : array size for stable_sort and rank (loop versions are O(n2))
    k       : number of top elements for topk
    """
    rng = np.random.default_rng(seed)

    arr_small  = rng.standard_normal(n_small)
    arr_large  = rng.standard_normal(n_large)
    arr_sorted = np.sort(rng.standard_normal(n_large))
    x_target   = float(arr_sorted[n_large // 2])

    bm = Benchmark(
        f"sort_search & rank — vectorised vs Python loop "
        f"(n_large={n_large}, n_small={n_small})"
    )

    # stable_sort — bubble sort O(n2) vs np.sort
    bm.run("stable_sort (Python loop)",   _stable_sort_loop, arr_small,
           n_repeat=3, warmup=0)
    bm.run("stable_sort (vectorised)",    _stable_sort_vec,  arr_small)

    # topk — list sort vs argpartition
    bm.run("topk (Python loop)",          _topk_loop, arr_large, k,
           n_repeat=5, warmup=1)
    bm.run("topk (vectorised)",           _topk_vec,  arr_large, k)

    # binary_search — linear scan O(n) vs searchsorted O(log n)
    bm.run("binary_search (Python loop)", _linear_search_loop, arr_sorted, x_target,
           n_repeat=3, warmup=0)
    bm.run("binary_search (vectorised)",  _binary_search_vec,  arr_sorted, x_target)

    # rank — O(n2) loop vs argsort O(n log n)
    bm.run("rank (Python loop)",          _rank_loop, arr_small,
           n_repeat=3, warmup=0)
    bm.run("rank (vectorised)",           _rank_vec,  arr_small)

    # percentile — Python sorted() vs np.sort + index
    bm.run("percentile (Python loop)",    _percentile_loop, arr_large, 50.0,
           n_repeat=5, warmup=1)
    bm.run("percentile (vectorised)",     _percentile_vec,  arr_large, 50.0)

    return bm


def bench_utils_loop_vs_vectorised(
    n_samples: int = 5_000,
    seed: int = 42,
) -> Benchmark:

    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n_samples)
    b = rng.standard_normal(n_samples)
    x = rng.standard_normal(n_samples)

    # sanity checks
    assert np.isclose(euclidean(a, b), _euclidean_loop(a, b)), \
        "euclidean mismatch"
    assert np.isclose(manhattan(a, b), _manhattan_loop(a, b)), \
        "manhattan mismatch"
    assert np.isclose(cosine_distance(a, b), _cosine_loop(a, b)), \
        "cosine mismatch"
    np.testing.assert_allclose(relu(x),       _relu_loop(x),       rtol=0,    atol=1e-15)
    np.testing.assert_allclose(leaky_relu(x), _leaky_relu_loop(x), rtol=0,    atol=1e-15)
    np.testing.assert_allclose(sigmoid(x),    _sigmoid_loop(x),    rtol=1e-12)
    np.testing.assert_allclose(tanh(x),       _tanh_loop(x),       rtol=1e-12)
    np.testing.assert_allclose(softmax(x),    _softmax_loop(x),    rtol=1e-8)
    np.testing.assert_allclose(
        logsumexp(x, axis=None), _logsumexp_loop(x), rtol=1e-8
    )

    bm = Benchmark(f"utils.py: vectorised vs Python loops (n={n_samples})")

    bm.run("euclidean (Python loop)",  _euclidean_loop,  a, b, n_repeat=3, warmup=0)
    bm.run("euclidean (vectorised)",   euclidean,        a, b)

    bm.run("manhattan (Python loop)",  _manhattan_loop,  a, b, n_repeat=3, warmup=0)
    bm.run("manhattan (vectorised)",   manhattan,        a, b)

    bm.run("cosine (Python loop)",     _cosine_loop,     a, b, n_repeat=3, warmup=0)
    bm.run("cosine (vectorised)",      cosine_distance,  a, b)

    bm.run("relu (Python loop)",       _relu_loop,       x, n_repeat=3, warmup=0)
    bm.run("relu (vectorised)",        relu,             x)

    bm.run("leaky_relu (Python loop)", _leaky_relu_loop, x, n_repeat=3, warmup=0)
    bm.run("leaky_relu (vectorised)",  leaky_relu,       x)

    bm.run("sigmoid (Python loop)",    _sigmoid_loop,    x, n_repeat=3, warmup=0)
    bm.run("sigmoid (vectorised)",     sigmoid,          x)

    bm.run("tanh (Python loop)",       _tanh_loop,       x, n_repeat=3, warmup=0)
    bm.run("tanh (vectorised)",        tanh,             x)

    bm.run("softmax (Python loop)",    _softmax_loop,    x, n_repeat=3, warmup=0)
    bm.run("softmax (vectorised)",     softmax,          x)

    bm.run("logsumexp (Python loop)",  _logsumexp_loop,  x, n_repeat=3, warmup=0)
    bm.run("logsumexp (vectorised)",   logsumexp,        x, axis=None)

    return bm



# Speedup reporters

def print_loop_vs_vec_speedups(bm: Benchmark, pairs: List[tuple]) -> None:
    print("\nSpeedup of vectorised over Python loop")
    print("-" * 50)
    for short, loop_label, vec_label in pairs:
        try:
            speedup = bm.compare(vec_label, loop_label)
            print(f"  {short:<16}: {speedup:>8.1f}x faster")
        except StopIteration:
            print(f"  {short:<16}: (missing labels in benchmark)")
    print()


def print_utils_speedups(bm: Benchmark) -> None:
    print("\nSpeedup of vectorised utils over Python loop")
    print("-" * 50)
    for short, loop_label, vec_label in _UTILS_PAIRS:
        try:
            speedup = bm.compare(vec_label, loop_label)
            print(f"  {short:<14}: {speedup:>8.1f}x faster")
        except StopIteration:
            print(f"  {short:<14}: (missing labels in benchmark)")
    print()



# run_all — runs every benchmark section

def run_all(n_samples: int = 100_000, seed: int = 42) -> Dict[str, Any]:

    # mean, accuracy, ssd
    loops_bm = bench_loops_vs_vectorised(n_samples=n_samples, seed=seed)
    loops_bm.print_report()
    print_loop_vs_vec_speedups(loops_bm, _LOOP_VS_VEC_PAIRS)

    # sort_search & rank
    sort_rank_bm = bench_sort_search_rank(
        n_large=n_samples,
        n_small=min(n_samples, 2_000),
        seed=seed,
    )
    sort_rank_bm.print_report()
    print_loop_vs_vec_speedups(sort_rank_bm, _SORT_RANK_PAIRS)

    # utils.py functions
    utils_bm = bench_utils_loop_vs_vectorised(n_samples=5_000, seed=seed)
    utils_bm.print_report()
    print_utils_speedups(utils_bm)

    return {
        "loops":     loops_bm,
        "sort_rank": sort_rank_bm,
        "utils":     utils_bm,
    }


if __name__ == "__main__":
    run_all(n_samples=100_000, seed=42)