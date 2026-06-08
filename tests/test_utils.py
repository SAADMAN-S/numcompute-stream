"""
test_preprocessing.py
Unit tests for utils: distances, activation functions, logsumexp, batching

Author: Risat Rahaman
"""

import numpy as np
import pytest
import sys

sys.path.append("../numcompute")

from numcompute.utils import (
    euclidean,
    manhattan,
    cosine_distance,
    relu,
    leaky_relu,
    sigmoid,
    tanh,
    softmax,
    logsumexp,
    batch,
)

class TestEuclidean:
    """Tests for euclidean() distance function."""

    def test_same_vectors(self):
        """Euclidean distance between identical vectors should be zero."""
        x = np.ones(5)

        assert np.allclose(euclidean(x, x), 0.0)

    def test_known_result_1d(self):
        """Check a simple known case: (3,0) and (0,4) should give 5.0."""
        x = np.array([3.0, 0.0])
        y = np.array([0.0, 4.0])

        assert np.allclose(euclidean(x, y), 5.0)

    def test_broadcasting(self):
        """Verify correct broadcasting: (2,2) against (2,) along last axis."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0.0, 0.0])

        result = euclidean(x, y, axis=-1)
        expected = np.sqrt(np.array([1**2 + 2**2, 3**2 + 4**2]))
        assert np.allclose(result, expected)

    def test_axis(self):
        """Test distance computation along a specified axis (axis=0)."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[0.0, 1.0], [2.0, 3.0]])
        
        result = euclidean(x, y, axis=0)
        expected = np.sqrt([2.0, 2.0])

        assert np.allclose(result, expected, atol=1e-6)

    def test_incompatible_shapes_raises(self):
        """Incompatible shapes should raise a ValueError."""
        with pytest.raises(ValueError):
            euclidean(np.ones((3, 4)), np.ones((5, 4)))

class TestManhattan:
    """Tests for manhattan() distance function."""

    def test_same_vectors(self):
        """Manhattan distance between identical vectors is zero."""
        x = np.ones(5)

        assert np.allclose(manhattan(x, x), 0.0)

    def test_known_result_1d(self):
        """Manhattan distance between (3,0) and (0,4) is 3+4 = 7."""
        x = np.array([3.0, 0.0])
        y = np.array([0.0, 4.0])

        assert np.allclose(manhattan(x, y), 7.0)

    def test_broadcasting_and_axis(self):
        """Test broadcasting and axis handling for Manhattan distance."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0.0, 0.0])

        result = manhattan(x, y, axis=-1)
        expected = np.array([3.0, 7.0])

        assert np.allclose(result, expected)

    def test_incompatible_shapes_raises(self):
        """Incompatible shapes should raise a ValueError."""
        with pytest.raises(ValueError):
            manhattan(np.ones((3, 4)), np.ones((5, 4)))


class TestCosineDistance:
    """Tests for cosine_distance() function."""

    def test_identical_vectors(self):
        """Cosine distance between identical vectors is zero."""
        x = np.array([1.0, 2.0, 3.0])

        assert np.allclose(cosine_distance(x, x), 0.0)

    def test_orthogonal_vectors(self):
        """Cosine distance of orthogonal vectors is 1.0."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])

        assert np.allclose(cosine_distance(x, y), 1.0)

    def test_opposite_vectors(self):
        """Cosine distance of opposite vectors is 2.0."""
        x = np.array([1.0, 0.0])
        y = np.array([-1.0, 0.0])

        assert np.allclose(cosine_distance(x, y), 2.0)

    def test_with_eps_zero_vector(self):
        """The eps parameter prevents division by zero for zero vectors."""
        x = np.zeros(5)
        y = np.ones(5)
        
        assert np.allclose(cosine_distance(x, y, eps=1e-10), 1.0)

    def test_broadcasting(self):
        """Test broadcasting behavior along a given axis."""
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([1.0, 1.0])

        expected = 1 - 1/np.sqrt(2)
        assert np.allclose(cosine_distance(x, y, axis=-1), expected)

    def test_incompatible_shapes_raises(self):
        """Incompatible input shapes should raise a ValueError."""
        with pytest.raises(ValueError):
            cosine_distance(np.ones(3), np.ones(4))


class TestRelu:
    """Tests for rectified linear unit (ReLU) activation."""

    def test_positive(self):
        """ReLU returns the input for positive values."""
        assert relu(5) == 5
        assert np.allclose(relu([1.0, 2.0]), [1.0, 2.0])

    def test_negative(self):
        """ReLU returns zero for negative values."""
        assert relu(-5) == 0
        assert np.allclose(relu([-1.0, 0.0]), [0.0, 0.0])

    def test_zero(self):
        """ReLU of zero is zero."""
        assert relu(0) == 0

    def test_array_type(self):
        """Output should be an ndarray of float type."""
        out = relu([-1, 2])
        assert isinstance(out, np.ndarray)
        assert out.dtype == float


class TestLeakyRelu:
    """Tests for leaky ReLU activation."""

    def test_positive(self):
        """Leaky ReLU returns the same value for positive inputs."""
        assert leaky_relu(5) == 5

    def test_negative_default_alpha(self):
        """Negative input with default alpha=0.01 returns 0.01 * input."""
        assert leaky_relu(-100) == -1.0  # 0.01 * -100

    def test_custom_alpha(self):
        """Negative input with a custom alpha value scales accordingly."""
        assert leaky_relu(-10, alpha=0.1) == -1.0

    def test_zero(self):
        """Leaky ReLU of zero is zero."""
        assert leaky_relu(0) == 0


class TestSigmoid:
    """Tests for sigmoid activation function."""

    def test_midpoint(self):
        """Sigmoid(0) should be 0.5."""
        assert np.allclose(sigmoid(0), 0.5)

    def test_large_positive(self):
        """Sigmoid of a large positive number approaches 1."""
        assert np.allclose(sigmoid(100), 1.0)

    def test_large_negative(self):
        """Sigmoid of a large negative number approaches 0."""
        assert np.allclose(sigmoid(-100), 0.0)

    def test_array(self):
        """Sigmoid applied to an array matches the elementwise formula."""
        out = sigmoid(np.array([-1, 0, 1]))
        expected = 1 / (1 + np.exp(-np.array([-1, 0, 1])))

        assert np.allclose(out, expected)


class TestTanh:
    """Tests for hyperbolic tangent activation."""

    def test_zero(self):
        """tanh(0) is zero."""
        assert np.allclose(tanh(0), 0)

    def test_large_positive(self):
        """tanh of a large positive number approaches 1."""
        assert np.allclose(tanh(10), 1.0, atol=1e-4)

    def test_large_negative(self):
        """tanh of a large negative number approaches -1."""
        assert np.allclose(tanh(-10), -1.0, atol=1e-4)

class TestSoftmax:
    """Tests for softmax probability computation."""

    def test_output_sums_to_one(self):
        """Softmax output sums to 1 across the specified axis."""
        x = np.random.randn(10)
        sm = softmax(x)

        assert np.allclose(sm.sum(), 1.0)
        assert np.all(sm >= 0)

    def test_stability(self):
        """Large values do not cause NaN or overflow."""
        large = np.array([1000.0, 1000.0])
        sm = softmax(large)

        assert not np.any(np.isnan(sm))
        assert np.allclose(sm.sum(), 1.0)

    def test_handles_negative_large(self):
        """Very negative values still return valid probabilities."""
        x = np.array([-1000.0, -1000.0])
        sm = softmax(x)

        assert np.allclose(sm, [0.5, 0.5])


class TestLogsumexp:
    """Tests for numerically stable logsumexp."""

    def test_single_value(self):
        """logsumexp of a single value returns that value."""
        assert np.allclose(logsumexp(np.array([1.0])), 1.0)

    def test_identical_values(self):
        """logsumexp of N identical values is value + log(N)."""
        N = 5
        v = 3.0
        x = np.full(N, v)
        expected = v + np.log(N)

        assert np.allclose(logsumexp(x), expected)

    def test_axis(self):
        """logsumexp works correctly along different axes."""
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        # axis=0:
        res = logsumexp(x, axis=0)
        expected0 = np.log(np.exp([0, 2]) + np.exp([1, 3]))
        assert np.allclose(res, expected0)
        # axis=1:
        res = logsumexp(x, axis=1)
        expected1 = np.array([np.log(np.exp(0) + np.exp(1)),
                              np.log(np.exp(2) + np.exp(3))])
        assert np.allclose(res, expected1)

    def test_keepdims(self):
        """keepdims=True squeezes the result, removing the reduced axis."""
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        res = logsumexp(x, axis=0, keepdims=True)
        # shape (1,2) -> squeezed to (2,)
        assert res.shape == (2,)
        assert np.allclose(res, logsumexp(x, axis=0))

        res2 = logsumexp(x, axis=1, keepdims=True)
        assert res2.shape == (2,)
        assert np.allclose(res2, logsumexp(x, axis=1))

        # keepdims=False also returns squeezed
        res3 = logsumexp(x, axis=1, keepdims=False)
        assert res3.shape == (2,)
        assert np.allclose(res3, logsumexp(x, axis=1))

    def test_stability_large_values(self):
        """Large values do not cause inf or nan."""
        x = np.array([1000.0, 1000.0])
        res = logsumexp(x)

        assert np.isfinite(res)
        assert np.allclose(res, 1000.0 + np.log(2))

    def test_axis_none(self):
        """When axis is None, logsumexp reduces all elements to a scalar."""
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        res = logsumexp(x)
        assert np.isscalar(res) or res.ndim == 0

        expected = np.log(np.sum(np.exp(x)))
        assert np.allclose(res, expected)

class TestSoftmax:
    """Tests for softmax probability computation."""

    def test_output_sums_to_one(self):
        """Softmax output sums to 1 across the specified axis."""
        x = np.random.randn(10)
        sm = softmax(x)

        assert np.allclose(sm.sum(), 1.0)
        assert np.all(sm >= 0)

    def test_stability(self):
        """Large values do not cause NaN or overflow."""
        large = np.array([1000.0, 1000.0])
        sm = softmax(large)

        assert not np.any(np.isnan(sm))
        assert np.allclose(sm.sum(), 1.0)

    def test_handles_negative_large(self):
        """Very negative values still return valid probabilities."""
        x = np.array([-1000.0, -1000.0])
        sm = softmax(x)

        assert np.allclose(sm, [0.5, 0.5])


class TestLogsumexp:
    """Tests for numerically stable logsumexp."""

    def test_single_value(self):
        """logsumexp of a single value returns that value."""
        assert np.allclose(logsumexp(np.array([1.0])), 1.0)

    def test_identical_values(self):
        """logsumexp of N identical values is value + log(N)."""
        N = 5
        v = 3.0
        x = np.full(N, v)
        expected = v + np.log(N)
        assert np.allclose(logsumexp(x), expected)     

    def test_keepdims(self):
        """keepdims=True squeezes the result, removing the reduced axis."""
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        res = logsumexp(x, axis=0, keepdims=True)
  
        assert res.shape == (2,)
        assert np.allclose(res, logsumexp(x, axis=0))

        res2 = logsumexp(x, axis=1, keepdims=True)
        assert res2.shape == (2,)
        assert np.allclose(res2, logsumexp(x, axis=1))

        res3 = logsumexp(x, axis=1, keepdims=False)
        assert res3.shape == (2,)
        assert np.allclose(res3, logsumexp(x, axis=1))

    def test_stability_large_values(self):
        """Large values do not cause inf or nan."""
        x = np.array([1000.0, 1000.0])
        res = logsumexp(x)

        assert np.isfinite(res)
        assert np.allclose(res, 1000.0 + np.log(2))

    def test_axis_none(self):
        """When axis is None, logsumexp reduces all elements to a scalar."""
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        res = logsumexp(x)
        assert np.isscalar(res) or res.ndim == 0

        expected = np.log(np.sum(np.exp(x)))
        assert np.allclose(res, expected)


class TestBatch:
    """Tests for batch generator utility."""

    def test_single_array(self):
        a = np.arange(10)
        batches = list(batch(a, batch_size=3, shuffle=False))

        assert len(batches) == 4  # 3+3+3+1
        assert np.array_equal(batches[0][0], [0, 1, 2])
        assert np.array_equal(batches[-1][0], [9])

    def test_multiple_arrays(self):
        """Multiple arrays are yielded in sync."""
        a = np.arange(6)
        b = -np.arange(6)

        batches = list(batch(a, b, batch_size=2, shuffle=False))

        assert len(batches) == 3
        
        for (aa, bb) in batches:
            assert np.array_equal(aa, -bb)

    def test_shuffle_off(self):
        """When shuffle=False, batches concatenate back to the original order."""
        a = np.arange(4)
        batches = list(batch(a, batch_size=2, shuffle=False))
        flat = np.concatenate([b[0] for b in batches])

        assert np.array_equal(flat, a)

    def test_shuffle_reproducibility(self):
        """Fixed seed makes shuffling deterministic."""
        a = np.arange(10)
        batches1 = list(batch(a, batch_size=3, shuffle=True, seed=42))
        batches2 = list(batch(a, batch_size=3, shuffle=True, seed=42))

        flat1 = np.concatenate([b[0] for b in batches1])
        flat2 = np.concatenate([b[0] for b in batches2])

        assert np.array_equal(flat1, flat2)

    def test_empty_arrays_raises(self):
        """Passing no arrays raises a ValueError."""
        with pytest.raises(ValueError, match="At least one array"):
            list(batch(batch_size=2))

    def test_different_first_dim_raises(self):
        """If arrays have different first dimension, a ValueError is raised."""
        a = np.ones(5)
        b = np.ones(6)
        with pytest.raises(ValueError, match="same first dimension"):
            list(batch(a, b, batch_size=2))

    def test_batch_size_larger_than_data(self):
        """If batch_size > number of samples, one full batch is returned."""
        a = np.arange(5)
        batches = list(batch(a, batch_size=10, shuffle=False))

        assert len(batches) == 1
        assert np.array_equal(batches[0][0], a)

    def test_yields_tuples(self):
        """Each yielded item is a tuple of array slices."""
        a = np.arange(3)
        gen = batch(a, batch_size=2, shuffle=False)
        first = next(gen)

        assert isinstance(first, tuple)
        assert len(first) == 1
