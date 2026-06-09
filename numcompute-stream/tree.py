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
        Maximum depth of the tree. None means nodes are expanded until all, leaves are pure or contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    criterion : {'gini', 'entropy'}, default='gini', Function used to measure split quality.
    max_features : int, float, {'sqrt', 'log2'}, or None, default=None
        The features we have consider at each split and their corresponding data types
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

    
    # Building a tree

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """Recursively build the tree and return the root node."""
        node = _Node(depth=depth)
 
        n_samples = X.shape[0]
        unique_classes = np.unique(y)
 
        if (
            n_samples < self.min_samples_split
            or len(unique_classes) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            node.is_leaf = True
            node.value = self._majority_class(y)
            return node
 
        feature, threshold, gain = self._best_split(X, y)
 
        if feature is None or gain <= 0.0:
            node.is_leaf = True
            node.value = self._majority_class(y)
            return node
 
        node.feature_index = feature
        node.threshold = threshold
 
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
 
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            node.is_leaf = True
            node.value = self._majority_class(y)
            return node
 
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node
 
    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Find the best feature_index, threshold split.
        Returns best_feature : int or None, best_threshold : float or None, best_gain : float
        """
        n_samples, n_features = X.shape
        current_impurity = self._impurity(y)
 
        best_gain = 0.0
        best_feature = None
        best_threshold = None
 
        n_consider = self._n_features_to_consider(n_features)
        feature_indices = np.random.choice(
            n_features, size=n_consider, replace=False)
 
        for fi in feature_indices:
            values = X[:, fi]
            unique_vals = np.unique(values)
 
            if unique_vals.size < 2:
                continue
 
            # Midpoints between consecutive unique values
            candidates = (unique_vals[:-1] + unique_vals[1:]) / 2.0
 
            # Vectorised gain computation
            for thresh in candidates:
                left_mask = values <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
 
                if n_left == 0 or n_right == 0:
                    continue
 
                gain = current_impurity - (
                    n_left  / n_samples * self._impurity(y[left_mask])
                    + n_right / n_samples * self._impurity(y[~left_mask])
                )
 
                if gain > best_gain:
                    best_gain = gain
                    best_feature = fi
                    best_threshold = thresh
 
        return best_feature, best_threshold, best_gain
 
    # Prediction helpers 
    def _predict_one(self, x: np.ndarray, node: _Node):
        """Traverse the tree and return the predicted class for one sample."""
        if node.is_leaf:
            return node.value
 
        feature_val = x[node.feature_index]
        if np.isnan(feature_val):
            return self._predict_one(x, node.left)
 
        if feature_val <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)
 
    # Impurity measures
    def _impurity(self, y: np.ndarray) -> float:
        """Compute Gini or entropy impurity for a label array."""
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / y.size
 
        if self.criterion == "gini":
            return float(1.0 - np.sum(probs ** 2))
        # entropy
        return float(-np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0))))
 
    # Utilities
    @staticmethod
    def _majority_class(y: np.ndarray):
        """Return the most frequent class in y."""
        classes, counts = np.unique(y, return_counts=True)
        return classes[int(np.argmax(counts))]
 
    def _n_features_to_consider(self, n_features: int) -> int:
        """Resolve max_features to a concrete integer."""
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, int):
            return max(1, min(mf, n_features))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(max(n_features, 2))))
        raise ValueError(
            f"max_features must be int, float, 'sqrt', 'log2', or None. Got {mf!r}.")
 
    @staticmethod
    def _validate_Xy(X, y):
        """Validate and convert X and y."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}.")
        return X, y
 
    @staticmethod
    def _fill_nan(X: np.ndarray) -> np.ndarray:
        """Replace NaN values with per-column means (NaN-safe)."""
        if not np.any(np.isnan(X)):
            return X
        X = X.copy()
        col_means = np.nanmean(X, axis=0)
        # If an entire column is NaN, fill with 0
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        return X
 
    def _check_fitted(self):
        """Raise if the tree has not been fitted yet."""
        if self.root is None:
            raise RuntimeError(
                "Tree has not been fitted. Call fit() or partial_fit() first.")