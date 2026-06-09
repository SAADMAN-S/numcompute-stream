"""
Unit tests for numcompute/metrics.py.
pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream import metrics


# Accuracy

def test_accuracy_perfect():
    y_true = [1, 0, 1]
    y_pred = [1, 0, 1]
    assert metrics.accuracy(y_true, y_pred) == 1.0


def test_accuracy_partial():
    y_true = [1, 0, 1]
    y_pred = [1, 1, 0]
    assert metrics.accuracy(y_true, y_pred) == pytest.approx(1/3)


def test_accuracy_mismatched_length():
    with pytest.raises(ValueError):
        metrics.accuracy([1, 2], [1])



# Confusion Matrix

def test_confusion_matrix_basic():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]

    cm = metrics.confusion_matrix(y_true, y_pred)

    assert cm.shape == (2, 2)
    assert cm.sum() == 4


def test_confusion_matrix_labels():
    y_true = [2, 1, 2]
    y_pred = [2, 2, 1]

    cm = metrics.confusion_matrix(y_true, y_pred, labels=[1, 2])
    assert cm.shape == (2, 2)



# Precision

def test_precision_binary():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]

    p = metrics.precision(y_true, y_pred)
    assert p == pytest.approx(0.5)


def test_precision_zero_division():
    y_true = [0, 0]
    y_pred = [0, 0]

    p = metrics.precision(y_true, y_pred)
    assert p == 0.0


def test_precision_macro():
    y_true = [0, 1, 2]
    y_pred = [0, 2, 1]

    p = metrics.precision(y_true, y_pred, average="macro")
    assert 0 <= p <= 1


def test_precision_per_class():
    y_true = [0, 1, 2]
    y_pred = [0, 2, 1]

    p = metrics.precision(y_true, y_pred, average=None)
    assert isinstance(p, np.ndarray)
    assert len(p) == 3



# Recall

def test_recall_binary():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]

    r = metrics.recall(y_true, y_pred)
    assert r == pytest.approx(0.5)


def test_recall_macro():
    y_true = [0, 1, 2]
    y_pred = [0, 2, 1]

    r = metrics.recall(y_true, y_pred, average="macro")
    assert 0 <= r <= 1


def test_recall_per_class():
    y_true = [0, 1, 2]
    y_pred = [0, 2, 1]

    r = metrics.recall(y_true, y_pred, average=None)
    assert isinstance(r, np.ndarray)
    assert len(r) == 3



# F1 Score

def test_f1_binary():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]

    f1 = metrics.f1(y_true, y_pred)
    assert f1 == pytest.approx(0.5)


def test_f1_macro():
    y_true = [0, 1, 2]
    y_pred = [0, 2, 1]

    f1 = metrics.f1(y_true, y_pred, average="macro")
    assert 0 <= f1 <= 1


def test_f1_zero_division():
    y_true = [0, 0]
    y_pred = [0, 0]

    f1 = metrics.f1(y_true, y_pred)
    assert f1 == 0.0


# MSE

def test_mse_basic():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 4]

    assert metrics.mse(y_true, y_pred) == pytest.approx(1/3)


def test_mse_zero():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]

    assert metrics.mse(y_true, y_pred) == 0.0



# ROC Curve + AUC

def test_roc_curve_basic():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)

    assert len(fpr) == len(tpr)
    assert len(thresholds) >= 2


def test_auc_range():
    x = [0, 0.5, 1]
    y = [0, 0.75, 1]

    area = metrics.auc(x, y)
    assert 0 <= area <= 1


def test_roc_invalid_multiclass():
    y_true = [0, 1, 2]
    y_score = [0.1, 0.2, 0.3]

    with pytest.raises(ValueError):
        metrics.roc_curve(y_true, y_score)



# Edge Cases

def test_empty_input():
    with pytest.raises(ValueError):
        metrics.accuracy([], [])


def test_all_wrong_predictions():
    y_true = [1, 1, 1]
    y_pred = [0, 0, 0]

    assert metrics.accuracy(y_true, y_pred) == 0.0
