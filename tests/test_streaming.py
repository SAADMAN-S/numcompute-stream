"""
test_streaming.py

Unit tests for streaming extensions of existing NumCompute modules:
- StreamingMetrics (metrics.py)
- StandardScaler.partial_fit, OneHotEncoder.partial_fit, SimpleImputer.partial_fit (preprocessing.py)
- StreamingStats (stats.py)
- Pipeline.partial_fit (pipeline.py)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from numcompute_stream.metrics import StreamingMetrics
from numcompute_stream.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from numcompute_stream.stats import StreamingStats
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture
def simple_dataset():
    X = np.array([
        [1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [2.5, 2.0],
        [5.0, 6.0], [6.0, 5.0], [5.5, 5.5], [6.5, 6.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


class TestStreamingMetrics:

    def test_update_and_result(self):
        """Accumulates correctly over two chunks."""
        sm = StreamingMetrics()
        sm.update([0, 1, 1, 0], [0, 1, 0, 0])
        sm.update([1, 0, 1, 1], [1, 0, 1, 0])
        result = sm.result()
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_reset_clears_state(self):
        """reset() wipes accumulated confusion matrix."""
        sm = StreamingMetrics()
        sm.update([0, 1], [0, 1])
        sm.reset()
        with pytest.raises(RuntimeError):
            sm.result()

    def test_rolling_window(self):
        """rolling_result() uses only the last window_size chunks."""
        sm = StreamingMetrics(window_size=2)
        sm.update([0, 0], [0, 0])
        sm.update([1, 1], [1, 1])
        sm.update([0, 0], [1, 1])
        assert sm.rolling_result()["accuracy"] < 1.0

    def test_no_window_size_rolling_raises(self):
        """rolling_result() raises if window_size not set."""
        sm = StreamingMetrics()
        sm.update([0, 1], [0, 1])
        with pytest.raises(RuntimeError):
            sm.rolling_result()

    def test_empty_raises(self):
        """result() before any update raises RuntimeError."""
        with pytest.raises(RuntimeError):
            StreamingMetrics().result()

    def test_n_chunks_increments(self):
        """n_chunks matches number of update() calls."""
        sm = StreamingMetrics()
        for _ in range(4):
            sm.update([0, 1], [0, 1])
        assert sm.result()["n_chunks"] == 4


class TestStandardScalerPartialFit:

    def test_partial_fit_single_chunk(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sc = StandardScaler()
        sc.partial_fit(X)
        assert sc.mean is not None
        assert sc.mean.shape == (2,)

    def test_partial_fit_incremental_mean(self):
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        sc = StandardScaler()
        sc.partial_fit(X1)
        sc.partial_fit(X2)
        expected = np.mean(np.vstack([X1, X2]), axis=0)
        np.testing.assert_allclose(sc.mean, expected, atol=1e-6)

    def test_partial_fit_zero_variance_safe(self):
        X = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
        sc = StandardScaler()
        sc.partial_fit(X)
        assert np.all(np.isfinite(sc.transform(X)))

    def test_partial_fit_then_transform(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        sc = StandardScaler()
        sc.partial_fit(X)
        X_t = sc.transform(X)
        assert X_t.shape == X.shape
        assert np.all(np.isfinite(X_t))


class TestOneHotEncoderPartialFit:

    def test_partial_fit_basic(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array([["cat"], ["dog"]]))
        assert "cat" in enc.categories[0]
        assert "dog" in enc.categories[0]

    def test_partial_fit_expands_categories(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array([["cat"], ["dog"]]))
        enc.partial_fit(np.array([["fish"]]))
        assert "fish" in enc.categories[0]
        assert len(enc.categories[0]) == 3


class TestSimpleImputerPartialFit:

    def test_partial_fit_noop(self):
        imp = SimpleImputer(fill_value=0)
        assert imp.partial_fit(np.array([[1.0, np.nan]])) is imp


class TestStreamingStats:

    def test_update_stats_basic(self):
        ss = StreamingStats()
        ss.update_stats([1.0, 2.0, 3.0])
        assert abs(ss.running_mean - 2.0) < 1e-10

    def test_update_stats_multiple_chunks(self):
        ss = StreamingStats()
        ss.update_stats([1.0, 2.0, 3.0])
        ss.update_stats([4.0, 5.0, 6.0])
        expected = np.mean([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert abs(ss.running_mean - expected) < 1e-10

    def test_nan_ignored(self):
        ss = StreamingStats()
        ss.update_stats([1.0, np.nan, 3.0])
        assert ss.n_samples == 2
        assert abs(ss.running_mean - 2.0) < 1e-10

    def test_reset(self):
        ss = StreamingStats()
        ss.update_stats([1.0, 2.0])
        ss.reset()
        assert ss._mean is None
        assert ss.n_samples == 0

    def test_windowed_histogram(self):
        ss = StreamingStats(bins=5, window_size=2)
        ss.update_stats([1.0, 2.0, 3.0])
        ss.update_stats([4.0, 5.0, 6.0])
        hist, edges = ss.get_histogram()
        assert hist.sum() == 6
        assert len(edges) == 6

    def test_summary_keys(self):
        ss = StreamingStats()
        ss.update_stats([1.0, 2.0, 3.0])
        for k in ("mean", "var", "std", "n_samples", "n_chunks"):
            assert k in ss.summary()


class TestPipelinePartialFit:

    def test_partial_fit_transformer_only(self):
        scaler = StandardScaler()
        pipe = Pipeline([("scale", scaler)])
        pipe.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert scaler.mean is not None

    def test_partial_fit_with_model(self, simple_dataset):
        X, y = simple_dataset
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3)),
        ])
        pipe.partial_fit(X[:4], y[:4])
        pipe.partial_fit(X[4:], y[4:])
        assert pipe.predict(X).shape == y.shape