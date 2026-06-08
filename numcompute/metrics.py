"""
metrics.py

Evaluation metrics for classification and regression tasks in NumCompute.

Author: Benzamin Yasir
"""
from __future__ import annotations

import numpy as np


def _as_1d_array(x, name: str) -> np.ndarray:
    """Convert input to a flattened 1D NumPy array."""
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr.ravel()


def _check_same_length(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Ensure y_true and y_pred have the same length."""
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length, got "
            f"{y_true.shape[0]} and {y_pred.shape[0]}."
        )
        
def _safe_divide(num, den, zero_division=0.0):
    """Safely divide, returning zero_division when denominator is zero."""
    den = np.asarray(den, dtype=float)
    num = np.asarray(num, dtype=float)

    result = np.full_like(num, fill_value=float(zero_division), dtype=float)
    mask = den != 0
    result[mask] = num[mask] / den[mask]
    return result


def accuracy(y_true, y_pred) -> float:
    """
    Parameters:-
    y_true : array-like of shape
        Ground truth labels.
    y_pred : array-like of shape
        Predicted labels.

    Returns:-
    float - Accuracy score in [0, 1].
    """
    y_true = _as_1d_array(y_true, "y_true")
    y_pred = _as_1d_array(y_pred, "y_pred")
    _check_same_length(y_true, y_pred)

    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """
    Compute confusion matrix.

    Parameters:-
    y_true : array-like of shape
        Ground truth labels.
    y_pred : array-like of shape
        Predicted labels.
    labels : array-like or None, default=None
        Label ordering. If None, inferred from sorted union of labels.

    Returns:-
    np.ndarray of shape
        Confusion matrix where rows are true labels and columns are predicted labels.
    """
    y_true = _as_1d_array(y_true, "y_true")
    y_pred = _as_1d_array(y_pred, "y_pred")
    _check_same_length(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    n_classes = len(labels)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        if t in label_to_index and p in label_to_index:
            cm[label_to_index[t], label_to_index[p]] += 1

    return cm


def precision(y_true, y_pred, average: str = "binary", pos_label=1, zero_division: float = 0.0):
    """
    Compute precision score.

    Parameters:-
    y_true : array-like
    y_pred : array-like
    average : {'binary', 'macro', 'weighted', None}, default='binary'
        Averaging method.
    pos_label : object, default=1
        Positive class for binary classification.
    zero_division : float, default=0.0
        Value to return when denominator is zero.

    Returns:-
    float or np.ndarray
    """
    y_true = _as_1d_array(y_true, "y_true")
    y_pred = _as_1d_array(y_pred, "y_pred")
    _check_same_length(y_true, y_pred)

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        return float(_safe_divide(tp, tp + fp, zero_division=zero_division))

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    support = np.sum(cm, axis=1)

    per_class = _safe_divide(tp, tp + fp, zero_division=zero_division)

    if average is None:
        return per_class
    if average == "macro":
        return float(np.mean(per_class))
    if average == "weighted":
        total = np.sum(support)
        if total == 0:
            return float(zero_division)
        return float(np.sum(per_class * support) / total)

    raise ValueError("average must be one of {'binary', 'macro', 'weighted', None}.")



def recall(y_true, y_pred, average: str = "binary", pos_label=1, zero_division: float = 0.0):
    """
    Compute recall score.
    """
    y_true = _as_1d_array(y_true, "y_true")
    y_pred = _as_1d_array(y_pred, "y_pred")
    _check_same_length(y_true, y_pred)

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        return float(_safe_divide(tp, tp + fn, zero_division=zero_division))

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp
    support = np.sum(cm, axis=1)

    per_class = _safe_divide(tp, tp + fn, zero_division=zero_division)

    if average is None:
        return per_class
    if average == "macro":
        return float(np.mean(per_class))
    if average == "weighted":
        total = np.sum(support)
        if total == 0:
            return float(zero_division)
        return float(np.sum(per_class * support) / total)

    raise ValueError("average must be one of {'binary', 'macro', 'weighted', None}.")



def f1(y_true, y_pred, average: str = "binary", pos_label=1, zero_division: float = 0.0):
    """
    Compute F1 score.
    """
    if average == "binary":
        p = precision(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=zero_division)
        r = recall(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=zero_division)
        denom = p + r
        return float(zero_division) if denom == 0 else float(2 * p * r / denom)

    p = precision(y_true, y_pred, average=None, zero_division=zero_division)
    r = recall(y_true, y_pred, average=None, zero_division=zero_division)
    denom = p + r
    per_class = np.where(denom == 0, zero_division, 2 * p * r / denom)

    y_true_arr = _as_1d_array(y_true, "y_true")
    y_pred_arr = _as_1d_array(y_pred, "y_pred")
    labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    support = np.sum(cm, axis=1)

    if average is None:
        return per_class
    if average == "macro":
        return float(np.mean(per_class))
    if average == "weighted":
        total = np.sum(support)
        if total == 0:
            return float(zero_division)
        return float(np.sum(per_class * support) / total)

    raise ValueError("average must be one of {'binary', 'macro', 'weighted', None}.")



def roc_curve(y_true, y_score, pos_label=1):
    """
    Compute ROC curve for binary classification.

    Parameters:-
    y_true : array-like of shape
        True binary labels.
    y_score : array-like of shape
        Scores or probabilities for the positive class.
    pos_label : object, default=1
        Positive class label.

    Returns:-
    fpr : np.ndarray
        False positive rates.
    tpr : np.ndarray
        True positive rates.
    thresholds : np.ndarray
        Thresholds corresponding to the curve.

    Raises:-
    ValueError
        If y_true does not contain exactly two classes.
    """
    y_true = _as_1d_array(y_true, "y_true")
    y_score = _as_1d_array(y_score, "y_score").astype(float)
    _check_same_length(y_true, y_score)

    unique_labels = np.unique(y_true)
    if unique_labels.size != 2:
        raise ValueError("roc_curve is defined only for binary classification.")

    y_true_bin = (y_true == pos_label).astype(int)

    desc_order = np.argsort(-y_score, kind="stable")
    y_score = y_score[desc_order]
    y_true_bin = y_true_bin[desc_order]

    thresholds = np.r_[np.inf, np.unique(y_score)[::-1]]

    P = np.sum(y_true_bin == 1)
    N = np.sum(y_true_bin == 0)

    if P == 0 or N == 0:
        raise ValueError("Both positive and negative samples are required.")

    tpr = [0.0]
    fpr = [0.0]

    for thresh in thresholds[1:]:
        y_pred = (y_score >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true_bin == 1))
        fp = np.sum((y_pred == 1) & (y_true_bin == 0))
        tpr.append(tp / P)
        fpr.append(fp / N)

    tpr.append(1.0)
    fpr.append(1.0)

    return np.asarray(fpr), np.asarray(tpr), thresholds



def auc(x, y) -> float:
    """
    Compute area under a curve

    Parameters:-
    x : array-like
        X coordinates.
    y : array-like
        Y coordinates.

    Returns:-
    float
        Area under the curve.
    """
    x = _as_1d_array(x, "x").astype(float)
    y = _as_1d_array(y, "y").astype(float)
    _check_same_length(x, y)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2.0
    return float(np.sum(dx * avg_y))

def mse(y_true, y_pred) -> float:
    """
    Compute mean squared error.

    Parameters:-
    y_true : array-like of shape
        Ground truth values.
    y_pred : array-like of shape
        Predicted values.

    Returns:-
    float
        Mean squared error.
    """
    y_true = _as_1d_array(y_true, "y_true").astype(float)
    y_pred = _as_1d_array(y_pred, "y_pred").astype(float)
    _check_same_length(y_true, y_pred)

    diff = y_true - y_pred
    return float(np.mean(diff ** 2))