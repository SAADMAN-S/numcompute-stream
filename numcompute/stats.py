"""
stats.py

Statistical functions for data analysis in NumCompute.

Author: Benzamin Yasir
"""
from __future__ import annotations

import numpy as np

def _as_array(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Convert input to a NumPy array and raise ValueError if the array is empty"""
    arr = np.asarray(X, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"'{name}' must not be empty.")
    return arr

def _normalize_axis(axis, ndim: int):
    """Validate axis argument"""
    if axis is None:
        return None
    if isinstance(axis, int):
        if axis < -ndim or axis >= ndim:
            raise np.exceptions.AxisError(axis, ndim=ndim)
        return axis
    raise TypeError("Axis must be None or an integer.")


def mean(x, axis=None, skipna: bool = True) -> np.ndarray:
    """
    Compute the mean of an array

    Parameters:-
    x : array-like
        Input data.
    axis : int or None, default=None
        Axis along which to compute the mean. If None, compute over all values.
    skipna : bool, default=True
        If True, ignores NaN values

    Returns:-
    np.ndarray or scalar
        Mean value(s).

    Raises:-
    ValueError
        If input is empty.
    TypeError
        If axis is invalid.

    Time Complexity:-
    O(n)

    Space Complexity:-
    O(1) extra (excluding output)
    """
    arr = _as_array(x)
    axis = _normalize_axis(axis, arr.ndim)
    fn = np.nanmean if skipna else np.mean
    return fn(arr, axis=axis)

def var(x, axis=None, ddof: int = 0, skipna: bool = True) -> np.ndarray:
    """Variance

    Parameters:-
    X : array-like
        Input data.
    axis : int or None, optional
        Reduction axis.
    ddof : int, default 0
        Delta degrees of freedom.
    skipna : bool, default True
        If True, ignores NaN values.

    Returns:-
    np.ndarray
        Variance value(s).

    Raises:-
    ValueError
        If X is empty.

    Complexity:-
    Time : O(n)   Space : O(1)
    """
    arr = _as_array(x)
    axis = _normalize_axis(axis, arr.ndim)
    fn = np.nanvar if skipna else np.var
    return fn(arr, axis=axis, ddof=ddof)

def std(x, axis=None, ddof: int = 0, skipna: bool = True) -> np.ndarray:
    """Standard deviation

    Parameters:-
    X : array-like
        Input data.
    axis : int or None, optional
        Reduction axis.
    ddof : int, default 0
        Delta degrees of freedom. Use 1 for sample std.
    skipna : bool, default True
        If True, ignores NaN values.

    Returns:-
    np.ndarray
        Standard deviation value(s).

    Raises:-
    ValueError
        If X is empty or ddof ≥ sample size along axis.

    Complexity:-
    Time : O(n)   Space : O(1)
    """
    arr = _as_array(x)
    axis = _normalize_axis(axis, arr.ndim)
    fn = np.nanstd if skipna else np.std
    return fn(arr, axis=axis, ddof=ddof)

def minimum(x, axis=None, skipna: bool = True):
    """Minimum value

    Parameters:-
    X : array-like
        Input data.
    axis : int or None, optional
        Reduction axis.
    skipna : bool, default True
        If True, ignores NaN values.

    Returns:-
    np.ndarray
        Minimum value(s).

    Raises:-
    ValueError
        If X is empty.

    Complexity:-
    Time : O(n)   Space : O(1)
    """
    arr = _as_array(x)
    axis = _normalize_axis(axis, arr.ndim)
    fn = np.nanmin if skipna else np.min
    return fn(arr, axis=axis)

def maximum(x, axis=None, skipna: bool = True):
    """Maximum value

    Parameters:-
    X : array-like
        Input data.
    axis : int or None, optional
        Reduction axis.
    skipna : bool, default True
        If True, ignores NaN values.

    Returns:-
    np.ndarray
        Maximum value(s).

    Raises:-
    ValueError
        If X is empty.

    Complexity:-
    Time : O(n)   Space : O(1)
    """
    arr = _as_array(x)
    axis = _normalize_axis(axis, arr.ndim)
    fn = np.nanmax if skipna else np.max
    return fn(arr, axis=axis)

def median(x, axis=None, skipna: bool = True):
    """Median value

    Parameters:-
    X : array-like
        Input data.
    axis : int or None, optional
        Reduction axis. 'None' flattens first.
    skipna : bool, default True
        If True, ignores NaN values

    Returns:-
    np.ndarray
        Median values.

    Raises:-
    ValueError
        If X is empty.

    Complexity:-
    Time : O(n log n)   Space : O(n) (sort-based)
    """
    arr = _as_array(x)
    axis = _normalize_axis(axis, arr.ndim)
    fn = np.nanmedian if skipna else np.median
    return fn(arr, axis=axis)

def histogram(x, bins=10, range=None, density: bool = False, skipna: bool = True):
    """
    Compute histogram counts and edges.

    Parameters:-
    x : array-like
        Input data. Flattened before histogram computation.
    bins : int or sequence, default=10
        Number of bins or explicit bin edges.
    range : tuple or None, default=None
        Lower and upper range of bins.
    density : bool, default=False
        If True, normalize histogram.

    Returns:-
    hist : np.ndarray
        Histogram counts or densities.
    bin_edges : np.ndarray
        Bin edges.
    """
    arr = _as_array(x)
    arr = arr.ravel()
    if skipna:
        arr = arr[~np.isnan(arr)] # ignore NaNs for histogram
    if arr.size == 0:
        raise ValueError("x contains only NaN values.")
    return np.histogram(arr, bins=bins, range=range, density=density)

def summary(x, axis=None, ddof: int = 1) -> dict:
    """Return a dict of common descriptive statistics.

    Parameters:-
    X : array-like, shape (...)
        Input data (NaNs are skipped in every statistic).
    axis : int or None, optional
        Reduction axis.
    ddof : int, default 1
        Delta degrees of freedom used for std/var.

    Returns:-
    dict
        Dictionary containing count, mean, std, min, median, max, var. NaN values are ignored in all statistics.
        
    Raises:-
    ValueError
        If X is empty.
    """
    arr = _as_array(x)
    ax = _normalize_axis(axis, arr.ndim)
    return {
        "mean":   np.nanmean(arr, axis=ax),
        "median": np.nanmedian(arr, axis=ax),
        "std":    np.nanstd(arr, axis=ax, ddof=ddof),
        "var":    np.nanvar(arr, axis=ax, ddof=ddof),
        "min":    np.nanmin(arr, axis=ax),
        "max":    np.nanmax(arr, axis=ax),
        "count":  np.sum(~np.isnan(arr), axis=ax),
    }


def quantile(
    X: np.ndarray,
    q,
    axis=None,
    interpolation: str = "linear",
    skipna: bool = True,
) -> np.ndarray:
    """Compute quantiles

    Parameters:-
    X : array-like
        Input data.
    q : float or array-like of float
        Quantile(s) in [0, 1].
    axis : int or None, optional
        Reduction axis.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        Interpolation method (passed to np.nanquantile / np.quantile).
    skipna : bool, default True
        If True, uses np.nanquantile.

    Returns:-
    np.ndarray
        Quantile value(s). Shape depends on q and axis.

    Raises:-
    ValueError
        If any value in q is outside [0, 1] or X is empty.

    Complexity:-
    Time : O(n log n)   Space : O(n)
    """
    X = _as_array(X)
    q = np.asarray(q, dtype=np.float64)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("All quantile values must be in the range [0, 1].")
    axis = _normalize_axis(axis, X.ndim)
    method_kw = {"method": interpolation}
    fn = np.nanquantile if skipna else np.quantile
    return fn(X, q, axis=axis, **method_kw)



class Welford:
    """
    Streaming mean/variance estimator using Welford's algorithm.

    Attributes:-
    n : int
        Number of observed samples.
    mean_ : float
        Running mean.
    M2_ : float
        Sum of squares of differences from the current mean.
    """

    def __init__(self):
        self.n = 0
        self.mean_ = 0.0
        self.M2_ = 0.0

    def update(self, x):
        """
        Update running statistics with one or more new observations.

        Parameters:-
        x : scalar or array-like
            New values to incorporate.
        """
        arr = np.asarray(x, dtype=float).ravel()
        for value in arr:
            if np.isnan(value):
                continue
            self.n += 1
            delta = value - self.mean_
            self.mean_ += delta / self.n
            delta2 = value - self.mean_
            self.M2_ += delta * delta2
        return self

    @property
    def variance(self):
        """Population variance."""
        if self.n == 0:
            return np.nan
        return self.M2_ / self.n

    @property
    def sample_variance(self):
        """Sample variance."""
        if self.n < 2:
            return np.nan
        return self.M2_ / (self.n - 1)

    @property
    def std(self):
        """Population standard deviation."""
        return np.sqrt(self.variance)

    @property
    def sample_std(self):
        """Sample standard deviation."""
        return np.sqrt(self.sample_variance)