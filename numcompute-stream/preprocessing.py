"""
preprocessing.py
Standard Scaler, Min Max Scaler and One Hot Encoder
"""

import numpy as np

class StandardScaler:
    """Standardise features by removing the mean and scaling to unit variance.

    Attributes:
        mean (ndarray or None): Per feature mean values computed during fitting.
        std (ndarray or None): Per feature standard deviation values computed
                               during fitting.
    """

    def __init__(self):
        """Initialize a StandardScaler with no stored statistics."""
        self.mean = None
        self.std = None

    def fit(self, X):
        """Compute the mean and standard deviation for each feature.

        Parameters:
            X: 2D numeric data, possibly containing NaN values.

        Returns:
            StandardScaler: Fitted scaler instance.
        """
        X_num = X.astype(float)

        self.mean = np.nanmean(X_num, axis=0)
        self.std = np.nanstd(X_num, axis=0)

        if np.ndim(self.std != 0):
            self.std[self.std == 0] = 1.0

        return self
    

    def partial_fit(self, X):
        """Incrementally update mean and std using Welford's batch algorithm.

        Parameters:
        X : array-like, shape (n_samples, n_features)
        Returns StandardScaler
        """
        X_num = np.asarray(X, dtype=float)
        if X_num.ndim == 1:
            X_num = X_num.reshape(1, -1)

        n_new = np.sum(~np.isnan(X_num), axis=0).astype(float)
        chunk_mean = np.nanmean(X_num, axis=0)
        chunk_var = np.nanvar(X_num, axis=0)

        if self.mean is None:
            self._n_seen = n_new
            self.mean = chunk_mean
            self._M2 = chunk_var * n_new
        else:
            n_old = self._n_seen
            n_total = n_old + n_new
            safe_total = np.where(n_total == 0, 1, n_total)
            delta = chunk_mean - self.mean
            self.mean = self.mean + delta * n_new / safe_total
            self._M2 = (self._M2 + chunk_var * n_new
                        + delta ** 2 * n_old * n_new / safe_total)
            self._n_seen = n_total

        var = np.where(self._n_seen > 1, self._M2 / self._n_seen, 0.0)
        self.std = np.sqrt(var)
        self.std[self.std == 0] = 1.0
        return self






    def transform(self, X):
        """Scale features using the stored mean and standard deviation.

        Parameters:
            X: 2D numeric data to be scaled.

        Returns:
            ndarray: Scaled data (zero mean and unit variance for features with
                     non-zero standard deviation).

        Raises:
            ValueError: If the scaler has not been fitted.
        """
        if self.mean is None:
            raise ValueError("StandardScaler has not been fitted yet. Call 'fit' first.")

        X_num = X.astype(float)

        return (X_num - self.mean) / self.std

    def fit_transform(self, X):
        """Fit the scaler and then transform the data.

        Parameters:
            X: 2D numeric data.

        Returns:
            ndarray: Scaled data.
        """
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale features to a given range (default [0, 1]).

    Attributes:
        min (ndarray or None): Per-feature minimum values computed during fitting.
        max (ndarray or None): Per-feature maximum values computed during fitting.
        feature_range (tuple): Target range (min, max) for scaled features.
    """

    def __init__(self, feature_range=(0, 1)):
        """Initialize a MinMaxScaler with a custom feature range.

        Parameters:
            feature_range (tuple): Desired output range (default (0, 1)).
        """
        self.feature_range = feature_range
        self.min = None
        self.max = None

    def fit(self, X):
        """Compute the minimum and maximum for each feature.

        Parameters:
            X: 2D numeric data, possibly containing NaN values.

        Returns:
            MinMaxScaler: Fitted scaler instance.
        """
        X_num = X.astype(float)
        self.min = np.nanmin(X_num, axis=0)
        self.max = np.nanmax(X_num, axis=0)

        range_mask = (self.max - self.min) == 0

        if np.ndim(X_num) == 1:
            if range_mask is True:
                self.max = self.min + 1.0
        else:
            self.max[range_mask] = self.min[range_mask] + 1.0

        return self

    def transform(self, X):
        """Scale features using the stored minimum and maximum.

        Parameters:
            X: 2D numeric data to be scaled.

        Returns:
            ndarray: Scaled data.

        Raises:
            ValueError: If the scaler has not been fitted.
        """
        if self.min is None or self.max is None:
            raise ValueError("MinMaxScaler has not been fitted yet. Call 'fit' first.")

        X_num = X.astype(float)
        min_val, max_val = self.feature_range
        X_std = (X_num - self.min) / (self.max - self.min)

        return X_std * (max_val - min_val) + min_val

    def fit_transform(self, X):
        """Fit the scaler and then transform the data.

        Parameters:
            X (array-like): 2D numeric data.

        Returns:
            ndarray: Scaled data.
        """
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Encode categorical features as a one-hot numeric array.

    Each column of the input is treated as a separate categorical feature. The
    encoder learns the unique categories for each column during fitting and
    produces as many binary columns as there are unique values per feature.

    Attributes:
        categories (list of ndarray or None): List where each element is a 1D
            array of unique categories found in the corresponding input column
            during fitting.
    """

    def __init__(self):
        """Initialize a OneHotEncoder with no stored categories."""
        self.categories = None

    def fit(self, X):
        """Determine the unique categories for each column.

        Parameters:
            X (array-like): 1D or 2D array of categorical data.

        Returns:
            OneHotEncoder: Fitted encoder.
        """
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        self.categories = [np.unique(X[:, i]) for i in range(n_features)]
        return self

    def partial_fit(self, X):
        """Incrementally expand known categories with new data.

        Parameters:
        X : array-like, shape (n_samples,) or (n_samples, n_features)

        Returns OneHotEncoder
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        if self.categories is None:
            self.categories = [np.unique(X[:, i]) for i in range(n_features)]
        else:
            if len(self.categories) != n_features:
                raise ValueError(
                    f"Expected {len(self.categories)} features, got {n_features}.")
            for i in range(n_features):
                new_cats = np.unique(X[:, i])
                self.categories[i] = np.unique(
                    np.concatenate([self.categories[i], new_cats]))
        return self

    def transform(self, X):
        """Encode the data using the fitted categories.

        Parameters:
            X: Categorical data to be encoded.

        Returns:
            ndarray: 2D one hot encoded array. Columns are ordered by feature,
                     and within each feature by the order of the stored categories.

        Raises:
            ValueError: If the encoder has not been fitted.
        """
        if self.categories is None:
            raise ValueError("OneHotEncoder has not been fitted yet. Call 'fit' first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        encoded_columns = []

        for i, cats in enumerate(self.categories):
            col_data = X[:, i]
            encoded = (col_data[:, None] == cats[None, :]).astype(float)
            encoded_columns.append(encoded)

        return np.hstack(encoded_columns)

    def fit_transform(self, X):
        """Fit the encoder and then transform the data.

        Parameters:
            X: Categorical data.

        Returns:
            ndarray: One hot encoded data.
        """
        return self.fit(X).transform(X)


class SimpleImputer:
    """Replace missing values with a constant.

    Attributes:
        fill_value: Constant used to replace missing values.
        missing_value: Placeholder to be replaced.
    """

    def __init__(self, fill_value=0, missing_value=np.nan):
        """Initialize a SimpleImputer.

        Parameters:
            fill_value: Value used to fill missing entries (default 0).
            missing_value: Placeholder value that indicates a missing entry (default np.nan).
        """
        self.fill_value = fill_value
        self.missing_value = missing_value

    def fit(self, X):
        """Fit the imputer.

        Parameters:
            X: Input data (unused, but required for API consistency).

        Returns:
            SimpleImputer.
        """
        return self
    def partial_fit(self, X):
        """No-op for constant imputer — satisfies streaming API.
        Parameters:
        X : array-like (unused)
        Returns SimpleImputer
        """
        return self
    def transform(self, X):
        """Replace missing values with the stored fill value.

        Parameters:
            X: Input data as a 2D numeric or string array.

        Returns:
            ndarray: Data with missing values replaced by 'fill_value'.
        """
        X_arr = np.asarray(X)

        if isinstance(self.missing_value, float) and np.isnan(self.missing_value):
            if not np.issubdtype(X_arr.dtype, np.floating):
                X_arr = X_arr.astype(float)

            mask = np.isnan(X_arr)

        else:
            mask = (X_arr == self.missing_value)

        return np.where(mask, self.fill_value, X_arr)

    def fit_transform(self, X):
        """Fit and transform in one step.

        Parameters:
            X: Input data.

        Returns:
            ndarray: Imputed data.
        """
        return self.fit(X).transform(X)

    