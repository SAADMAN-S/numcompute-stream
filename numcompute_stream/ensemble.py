"""
Ensemble classifiers built on top of DecisionTreeClassifier by implementing Random Forest and Bagging with streaming by partial_fit().
"""
from __future__ import annotations

import numpy as np
from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    """
    Ensemble classifier supporting Random Forest and Bagging methods.
    Manages N decision trees and accumulates their predictions via majority voting. 

    Parameters:
    n_estimators : int, default=10, Number of decision trees in the ensemble.
    method : {'random_forest', 'bagging'}, default='random_forest'
    -'random_forest': Each tree uses random feature subsets (max_features='sqrt')
      and bootstrap sampling.
    -'bagging': Each tree uses all features with bootstrap sampling.
    max_depth : int or None, default=None, Maximum depth for each individual tree.
    min_samples_split : int, default=2, Minimum samples required to split a node in each tree.
    criterion : {'gini', 'entropy'}, default='gini', Split condition passed to each tree.
    bootstrap : bool, default=True if to use bootstrap sampling when building each tree.
    
    max_features : int, float, {'sqrt', 'log2'}, or None, default=None
        Overrides the default feature sampling strategy.
        If None, defaults to 'sqrt' for random_forest and None for bagging.
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes:
    estimators_ : list of DecisionTreeClassifier
        Fitted individual trees.
    classes_ : np.ndarray
        Unique class labels seen during fitting.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        method: str = "random_forest",
        max_depth=None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        bootstrap: bool = True,
        max_features=None,
        random_state=None,
    ):
        if method not in ("random_forest", "bagging"):
            raise ValueError("method must be 'random_forest' or 'bagging'.")
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")

        self.n_estimators = n_estimators
        self.method = method
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state

        # Resolve max_features
        if max_features is not None:
            self.max_features = max_features
        elif method == "random_forest":
            self.max_features = "sqrt"
        else:
            self.max_features = None  # bagging uses all features

        self.estimators_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None

        # Accumulated data for partial_fit
        self._X_accum: np.ndarray | None = None
        self._y_accum: np.ndarray | None = None

        if random_state is not None:
            np.random.seed(random_state)

    # Public API
    def fit(self, X, y) -> "EnsembleClassifier":
        """
        Fit all trees on the training data. Each tree is trained on a sample of the data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns EnsembleClassifier as self
        """
        X, y = self._validate_Xy(X, y)
        self.classes_ = np.unique(y)
        self._X_accum = X.copy()
        self._y_accum = y.copy()

        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = self._make_tree()
            X_sample, y_sample = self._sample(X, y)
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

        return self

    def partial_fit(self, X_chunk, y_chunk) -> "EnsembleClassifier":
        """
        Incrementally update the ensemble with a new data chunk.
        Accumulates the chunk with all previous data and calls partial_fit on each individual tree using a bootstrap sample of the accumulated data.

        Parameters:
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)

        Returns as EnsembleClassifier
        """
        X_chunk, y_chunk = self._validate_Xy(X_chunk, y_chunk)

        if self._X_accum is None:
            self._X_accum = X_chunk
            self._y_accum = y_chunk
            self.estimators_ = [self._make_tree()
                                 for _ in range(self.n_estimators)]
        else:
            self._X_accum = np.vstack([self._X_accum, X_chunk])
            self._y_accum = np.concatenate([self._y_accum, y_chunk])

        self.classes_ = np.unique(self._y_accum)

        for tree in self.estimators_:
            X_sample, y_sample = self._sample(self._X_accum, self._y_accum)
            tree.partial_fit(X_sample, y_sample)

        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels via majority vote across all trees.
        Parameters:
        X : array-like, shape (n_samples, n_features)

        Returns np.ndarray, shape (n_samples,)
        Raises RuntimeError if the ensemble has not been fitted.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Collect predictions from all trees: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])

        # Majority vote per sample
        n_samples = X.shape[0]
        final_preds = np.empty(n_samples, dtype=all_preds.dtype)
        for i in range(n_samples):
            votes = all_preds[:, i]
            classes, counts = np.unique(votes, return_counts=True)
            final_preds[i] = classes[np.argmax(counts)]

        return final_preds

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities averaged across all trees.
        Parameters:
        X : array-like, shape (n_samples, n_features)
        Returns np.ndarray, shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba_sum = np.zeros((n_samples, n_classes), dtype=float)

        for tree in self.estimators_:
            tree_proba = tree.predict_proba(X)

            for j, cls in enumerate(tree.classes_):
                if cls in self.classes_:
                    idx = np.searchsorted(self.classes_, cls)
                    proba_sum[:, idx] += tree_proba[:, j]

        return proba_sum / self.n_estimators

    # Internal helpers

    def _make_tree(self) -> DecisionTreeClassifier:
        """Instantiate a new tree with this ensemble's configuration."""
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            max_features=self.max_features,
        )

    def _sample(self, X: np.ndarray, y: np.ndarray):
        """Return a sample of the data which could be possibly bootstrapped."""
        if not self.bootstrap:
            return X, y
        n = X.shape[0]
        indices = np.random.randint(0, n, size=n)
        return X[indices], y[indices]

    @staticmethod
    def _validate_Xy(X, y):
        """Validate and convert inputs."""
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

    def _check_fitted(self):
        """Raise if the ensemble has not been fitted."""
        if not self.estimators_:
            raise RuntimeError(
                "Ensemble has not been fitted. Call fit() or partial_fit() first.")