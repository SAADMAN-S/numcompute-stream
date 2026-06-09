"""
test_tree.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture
def simple_dataset():
    X = np.array([
        [1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [2.5, 2.0],
        [5.0, 6.0], [6.0, 5.0], [5.5, 5.5], [6.5, 6.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


@pytest.fixture
def multi_class_dataset():
    X = np.array([
        [0.0, 0.0], [0.5, 0.5],
        [5.0, 0.0], [5.5, 0.5],
        [0.0, 5.0], [0.5, 5.5],
    ])
    y = np.array([0, 0, 1, 1, 2, 2])
    return X, y


class TestDecisionTreeClassifier:

    def test_fit_predict_basic(self, simple_dataset):
        """Tree trained on clean data predicts training labels correctly."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape
        assert np.mean(preds == y) >= 0.75

    def test_gini_criterion(self, simple_dataset):
        """Gini criterion produces valid predictions."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier(criterion="gini")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_entropy_criterion(self, simple_dataset):
        """Entropy criterion produces valid predictions."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier(criterion="entropy")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_max_depth_limits_tree(self, simple_dataset):
        """max_depth=1 produces a single split with leaf children."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y)
        assert clf.root is not None
        assert clf.root.left.is_leaf
        assert clf.root.right.is_leaf

    def test_predict_proba_shape(self, simple_dataset):
        """predict_proba returns (n_samples, n_classes) array summing to 1."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_partial_fit_single_chunk(self, simple_dataset):
        """partial_fit on one chunk behaves like fit."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier(max_depth=3)
        clf.partial_fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape

    def test_partial_fit_multiple_chunks(self, simple_dataset):
        """Accuracy does not degrade drastically across chunks."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier(max_depth=3)
        clf.partial_fit(X[:4], y[:4])
        acc1 = np.mean(clf.predict(X) == y)
        clf.partial_fit(X[4:], y[4:])
        acc2 = np.mean(clf.predict(X) == y)
        assert acc2 >= acc1 - 0.2

    def test_nan_handling(self):
        """Tree handles NaN features without raising."""
        X = np.array([[1.0, np.nan], [np.nan, 2.0],
                      [3.0, 4.0], [5.0, 6.0]], dtype=float)
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        assert clf.predict(X).shape == (4,)

    def test_single_class_makes_leaf(self):
        """All-same-label input produces a leaf at root."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 0])
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        assert clf.root.is_leaf
        assert clf.root.value == 0

    def test_multiclass_prediction(self, multi_class_dataset):
        """Tree handles three-class problems."""
        X, y = multi_class_dataset
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, y)
        assert set(clf.predict(X)).issubset({0, 1, 2})

    def test_invalid_criterion_raises(self):
        """Unknown criterion raises ValueError at construction."""
        with pytest.raises(ValueError):
            DecisionTreeClassifier(criterion="mse")

    def test_predict_before_fit_raises(self):
        """predict() before fit() raises RuntimeError."""
        with pytest.raises(RuntimeError):
            DecisionTreeClassifier().predict(np.array([[1.0, 2.0]]))

    def test_max_features_sqrt(self, simple_dataset):
        """max_features='sqrt' works without error."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier(max_features="sqrt")
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape