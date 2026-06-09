import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream import stats



# Mean / Var / Std

def test_mean_basic():
    x = [1, 2, 3, 4]
    assert stats.mean(x) == 2.5


def test_mean_with_nan_skip():
    x = [1, np.nan, 3]
    assert stats.mean(x) == 2.0 


def test_mean_no_skipna():
    x = [1, np.nan, 3]
    result = stats.mean(x, skipna=False)
    assert np.isnan(result)


def test_var_basic():
    x = [1, 2, 3, 4]
    assert stats.var(x) == np.var(x)


def test_std_basic():
    x = [1, 2, 3, 4]
    assert stats.std(x) == np.std(x)



# Min / Max / Median

def test_minimum():
    x = [1, 2, 3]
    assert stats.minimum(x) == 1


def test_maximum():
    x = [1, 2, 3]
    assert stats.maximum(x) == 3


def test_median_even():
    x = [1, 2, 3, 4]
    assert stats.median(x) == 2.5


def test_median_with_nan():
    x = [1, np.nan, 3]
    assert stats.median(x) == 2.0


# Histogram

def test_histogram_basic():
    x = [1, 2, 2, 3]
    hist, edges = stats.histogram(x, bins=2)
    assert hist.sum() == 4


def test_histogram_ignore_nan():
    x = [1, 2, np.nan, 3]
    hist, _ = stats.histogram(x)
    assert hist.sum() == 3


def test_histogram_all_nan():
    x = [np.nan, np.nan]
    with pytest.raises(ValueError):
        stats.histogram(x)


# Summary

def test_summary_basic():
    x = [1, 2, 3, 4]
    result = stats.summary(x)

    assert result["mean"] == 2.5
    assert result["min"] == 1
    assert result["max"] == 4
    assert result["count"] == 4


def test_summary_with_nan():
    x = [1, np.nan, 3]
    result = stats.summary(x)

    assert result["mean"] == 2.0
    assert result["count"] == 2


# Quantile / Percentile / IQR

def test_quantile_basic():
    x = np.array([1, 2, 3, 4])
    assert stats.quantile(x, 0.5) == 2.5


def test_quantile_multiple():
    x = np.array([1, 2, 3, 4])
    q = stats.quantile(x, [0.25, 0.75])
    assert np.allclose(q, [1.75, 3.25])


def test_quantile_invalid():
    with pytest.raises(ValueError):
        stats.quantile(np.array([1, 2, 3]), 1.5)




# Axis Handling

def test_axis_mean():
    x = np.array([[1, 2], [3, 4]])
    result = stats.mean(x, axis=0)
    assert np.allclose(result, [2, 3])


def test_axis_invalid():
    x = np.array([[1, 2]])
    with pytest.raises(Exception):
        stats.mean(x, axis=5)



# Empty Input

def test_empty_input():
    with pytest.raises(ValueError):
        stats.mean([])



# Welford

def test_welford_mean_variance():
    x = [1, 2, 3, 4, 5]
    w = stats.Welford()
    w.update(x)

    assert np.isclose(w.mean_, np.mean(x))
    assert np.isclose(w.variance, np.var(x))


def test_welford_ignore_nan():
    x = [1, np.nan, 3]
    w = stats.Welford()
    w.update(x)

    assert np.isclose(w.mean_, 2.0)


def test_welford_empty():
    w = stats.Welford()
    assert np.isnan(w.variance)


def test_welford_sample_variance():
    x = [1, 2, 3, 4]
    w = stats.Welford()
    w.update(x)

    assert np.isclose(w.sample_variance, np.var(x, ddof=1))
