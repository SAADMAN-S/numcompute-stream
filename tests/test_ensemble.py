"""
test_ensemble.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from numcompute_stream.ensemble import EnsembleClassifier


@pytest.fixture
def simple_dataset():
    X = np.array([
        [1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [2.5, 2.0],
        [5.0, 6.0], [6.0, 5.0], [5.5, 5.5], [6.5, 6.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


class TestEnsembleClassifier:

    def test_random_forest_basic(self, simple_dataset):
        """Random forest trains and predicts without error."""
        X, y = simple_dataset
        rf = EnsembleClassifier(n_estimators=5, method="random_forest",
                                random_state=0)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert preds.shape == y.shape
        assert set(preds).issubset({0, 1})

    def test_bagging_basic(self, simple_dataset):
        """Bagging ensemble trains and predicts without error."""
        X, y = simple_dataset
        bag = EnsembleClassifier(n_estimators=5, method="bagging",
                                 random_state=0)
        bag.fit(X, y)
        assert bag.predict(X).shape == y.shape

    def test_n_estimators_correct(self, simple_dataset):
        """Ensemble creates exactly n_estimators trees."""
        X, y = simple_dataset
        rf = EnsembleClassifier(n_estimators=7, random_state=0)
        rf.fit(X, y)
        assert len(rf.estimators_) == 7

    def test_partial_fit_streaming(self, simple_dataset):
        """Ensemble partial_fit accumulates chunks correctly."""
        X, y = simple_dataset
        rf = EnsembleClassifier(n_estimators=5, random_state=0)
        rf.partial_fit(X[:4], y[:4])
        rf.partial_fit(X[4:], y[4:])
        assert rf.predict(X).shape == y.shape

    def test_predict_proba_sums_to_one(self, simple_dataset):
        """predict_proba rows sum to 1."""
        X, y = simple_dataset
        rf = EnsembleClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)
        proba = rf.predict_proba(X)
        assert proba.shape[1] == 2
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError):
            EnsembleClassifier(method="boosting")

    def test_predict_before_fit_raises(self, simple_dataset):
        """predict() before fit() raises RuntimeError."""
        X, _ = simple_dataset
        with pytest.raises(RuntimeError):
            EnsembleClassifier(n_estimators=3).predict(X)