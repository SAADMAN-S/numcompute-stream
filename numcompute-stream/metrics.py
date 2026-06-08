"""
metrics.py

Evaluation metrics for classification and regression tasks in NumCompute.

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


# Streaming metrics
class StreamingMetrics:
    """
    Incremental classification metrics with streaming update support.
    Maintains a running confusion matrix that accumulates across chunks.
    Additionally supports a rolling window over the last N chunks.
 
    Parameters: 
    labels : array-like or None. Class labels. Inferred from first update() call if None.
    window_size : int or None. If set, rolling_result() computes metrics over the last `window_size` chunks only.
    zero_division : float. Value returned when a denominator is zero.
    
    """
 
    def __init__(self, labels=None, window_size=None, zero_division=0.0):
        self._labels = None if labels is None else np.asarray(labels)
        self._window_size = window_size
        self._zero_division = zero_division
        self._cm = None
        self._chunk_buffer = [] 
        self._n_chunks = 0
 
    def update(self, y_true_chunk, y_pred_chunk) -> None:
        """
        Accumulate a new chunk into the running confusion matrix.
        Parameters:
        y_true_chunk : array-like, shape (n,)
        y_pred_chunk : array-like, shape (n,)
        """
        y_true_chunk = _as_1d_array(y_true_chunk, "y_true_chunk")
        y_pred_chunk = _as_1d_array(y_pred_chunk, "y_pred_chunk")
        _check_same_length(y_true_chunk, y_pred_chunk)
 
        new_labels = np.unique(np.concatenate([y_true_chunk, y_pred_chunk]))
        if self._labels is None:
            self._labels = new_labels
        else:
            all_labels = np.unique(np.concatenate([self._labels, new_labels]))
            if len(all_labels) > len(self._labels):
                n_new = len(all_labels)
                new_cm = np.zeros((n_new, n_new), dtype=int)
                old_idx = np.searchsorted(all_labels, self._labels)
                if self._cm is not None:
                    for i, oi in enumerate(old_idx):
                        for j, oj in enumerate(old_idx):
                            new_cm[oi, oj] = self._cm[i, j]
                self._cm = new_cm
                self._labels = all_labels
 
        chunk_cm = confusion_matrix(y_true_chunk, y_pred_chunk,labels=self._labels)
        self._cm = chunk_cm if self._cm is None else self._cm + chunk_cm
 
        if self._window_size is not None:
            self._chunk_buffer.append(
                (y_true_chunk.copy(), y_pred_chunk.copy()))
            if len(self._chunk_buffer) > self._window_size:
                self._chunk_buffer.pop(0)
 
        self._n_chunks = self._n_chunks + 1
 
    def result(self) -> dict:
        """
        Return metrics computed over all accumulated chunks.
        Gives: dict with keys: accuracy, precision_macro, recall_macro, f1_macro,confusion_matrix, n_chunks
        """
        if self._cm is None:
            raise RuntimeError("No data accumulated. Call update() first.")
        return self._compute_from_cm(self._cm, self._n_chunks)
 

    def rolling_result(self) -> dict:
        """
        Return metrics computed over the rolling window only.
        Raises RuntimeError if window_size was not set.
        """
        if self._window_size is None:
            raise RuntimeError(
                "window_size must be set at construction to use rolling_result().")
        if not self._chunk_buffer:
            raise RuntimeError("Rolling buffer is empty. Call update() first.")
 
        y_true_all = np.concatenate([b[0] for b in self._chunk_buffer])
        y_pred_all = np.concatenate([b[1] for b in self._chunk_buffer])
        cm = confusion_matrix(y_true_all, y_pred_all, labels=self._labels)
        return self._compute_from_cm(cm, len(self._chunk_buffer))
 
    def reset(self) -> None:
        """Reset all accumulated state where labels are preserved."""
        self._cm = None
        self._chunk_buffer = []
        self._n_chunks = 0
 
    def _compute_from_cm(self, cm: np.ndarray, n_chunks: int) -> dict:
        """Derive scalar metrics from a confusion matrix."""
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        total = int(np.sum(cm))
 
        acc = float(np.sum(tp) / total) if total > 0 else self._zero_division
 
        per_prec = _safe_divide(tp, tp + fp, zero_division=self._zero_division)
        per_rec = _safe_divide(tp, tp + fn, zero_division=self._zero_division)
        denom = per_prec + per_rec
        per_f1 = np.where(denom == 0, self._zero_division,
                          2 * per_prec * per_rec / denom)
 
        return {
            "accuracy": acc,
            "precision_macro": float(np.mean(per_prec)),
            "recall_macro": float(np.mean(per_rec)),
            "f1_macro": float(np.mean(per_f1)),
            "confusion_matrix": cm.copy(),
            "n_chunks": n_chunks,
        }