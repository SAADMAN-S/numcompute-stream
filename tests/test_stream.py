"""
test_stream.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture
def simple_dataset():
    X = np.array([
        [1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [2.5, 2.0],
        [5.0, 6.0], [6.0, 5.0], [5.5, 5.5], [6.5, 6.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


class TestStreamTrainer:

    def test_fit_chunk_logs_entry(self, simple_dataset):
        """fit_chunk appends one log entry per call."""
        X, y = simple_dataset
        trainer = StreamTrainer(model=DecisionTreeClassifier(max_depth=3))
        trainer.fit_chunk(X[:4], y[:4])
        trainer.fit_chunk(X[4:], y[4:])
        assert len(trainer.log_) == 2

    def test_accuracy_history_shape(self, simple_dataset):
        """accuracy_history length matches number of fit_chunk calls."""
        X, y = simple_dataset
        trainer = StreamTrainer(model=DecisionTreeClassifier(max_depth=3))
        for i in range(4):
            trainer.fit_chunk(X[i:i+1], y[i:i+1])
        assert trainer.accuracy_history().shape == (4,)

    def test_score_chunk_returns_accuracy(self, simple_dataset):
        """score_chunk returns a dict with accuracy in [0, 1]."""
        X, y = simple_dataset
        trainer = StreamTrainer(model=DecisionTreeClassifier(max_depth=3))
        trainer.fit_chunk(X[:4], y[:4])
        result = trainer.score_chunk(X[4:], y[4:])
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_score_chunk_before_fit_raises(self, simple_dataset):
        """score_chunk before any fit_chunk raises RuntimeError."""
        X, y = simple_dataset
        trainer = StreamTrainer(model=DecisionTreeClassifier())
        with pytest.raises(RuntimeError):
            trainer.score_chunk(X, y)

    def test_reset_clears_log(self, simple_dataset):
        """reset() clears the log and chunk counter."""
        X, y = simple_dataset
        trainer = StreamTrainer(model=DecisionTreeClassifier(max_depth=3))
        trainer.fit_chunk(X[:4], y[:4])
        trainer.reset()
        assert len(trainer.log_) == 0
        assert trainer._chunk_idx == 0

    def test_cumulative_metrics(self, simple_dataset):
        """cumulative_metrics returns valid dict after streaming."""
        X, y = simple_dataset
        trainer = StreamTrainer(model=DecisionTreeClassifier(max_depth=3))
        trainer.fit_chunk(X[:4], y[:4])
        trainer.fit_chunk(X[4:], y[4:])
        m = trainer.cumulative_metrics()
        assert "accuracy" in m
        assert m["n_chunks"] == 2