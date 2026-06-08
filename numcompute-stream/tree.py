"""
Decision tree classifier with streaming support. Supports Gini impurity and entropy split criteria.
"""
from __future__ import annotations
import numpy as np

#internal node structure

class _Node:
    """A single node in the decision tree.
 
    Attributes:
    feature_index : int or None
        Index of the feature used for splitting (None for leaf nodes).
    threshold : float or None
        Split threshold value (None for leaf nodes).
    left : _Node or None
        Left child (feature value <= threshold).
    right : _Node or None
        Right child (feature value > threshold).
    value : object or None
        Predicted class label (leaf nodes only).
    is_leaf : bool
        Whether this node is a terminal leaf.
    depth : int
        Depth of this node in the tree.
    """
 
    __slots__ = ("feature_index", "threshold", "left", "right",
                 "value", "is_leaf", "depth")
 
    def __init__(self, depth: int = 0):
        self.depth = depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = False


# Decision tree classifier
class DecisionTreeClassifier:
    """
    Depth-limited decision tree classifier built from scratch using NumPy.
    Parameters:
    max_depth : int or None, default=None
        Maximum depth of the tree. None means nodes are expanded until all
        leaves are pure or contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    criterion : {'gini', 'entropy'}, default='gini'
        Function used to measure split quality.
    max_features : int, float, {'sqrt', 'log2'}, or None, default=None
        Number of features to consider at each split.
        - int  : exact number of features
        - float: fraction of total features
        - 'sqrt': int(sqrt(n_features))
        - 'log2': int(log2(n_features))
        - None : all features
 
    Attributes:
    root : _Node or None
        Root node of the fitted tree.
    classes_ : np.ndarray
        Unique class labels seen during fit.
    n_features_ : int
        Number of features seen during fit.
    """
 
    def __init__(self, max_depth=None, min_samples_split: int = 2, criterion: str = "gini", max_features=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
 
        self.root: _Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None
    
        self._X_accum: np.ndarray | None = None
        self._y_accum: np.ndarray | None = None
 
    def fit(self, X, y) -> "DecisionTreeClassifier":
        """
        Build a decision tree from training data.
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training features. NaNs are replaced with column means.
        y : array-like, shape (n_samples,)
            Target class labels.
        Returns DecisionTreeClassifier as self
        """
        X, y = self._validate_Xy(X, y)
        X = self._fill_nan(X)
 
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self._X_accum = X.copy()
        self._y_accum = y.copy()
 
        self.root = self._build(X, y, depth=0)
        return self
    
    def partial_fit(self, X_chunk, y_chunk) -> "DecisionTreeClassifier":
        """
        Incrementally update the tree with a new data chunk and accumulates the new chunk with all previous data and rebuilds
        the tree from scratch on the full accumulated dataset.
 
        Parameters:
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)
        Returns DecisionTreeClassifier
        """
        X_chunk, y_chunk = self._validate_Xy(X_chunk, y_chunk)
 
        if self._X_accum is None:
            self._X_accum = X_chunk
            self._y_accum = y_chunk
        else:
            self._X_accum = np.vstack([self._X_accum, X_chunk])
            self._y_accum = np.concatenate([self._y_accum, y_chunk])
 
        return self.fit(self._X_accum, self._y_accum)
 

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels for samples in X.
        Parameters:
        X : array-like, shape (n_samples, n_features)
        Returns np.ndarray, shape (n_samples,)
        Raises RuntimeError if the tree has not been fitted.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = self._fill_nan(X)
        return np.array([self._predict_one(x, self.root) for x in X])

    

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
 
        Parameters:
        X : array-like, shape (n_samples, n_features)
 
        Returns np.ndarray, shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = self._fill_nan(X)
 
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        proba = np.zeros((X.shape[0], n_classes), dtype=float)
 
        for i, x in enumerate(X):
            pred = self._predict_one(x, self.root)
            if pred in class_to_idx:
                proba[i, class_to_idx[pred]] = 1.0
 
        return proba