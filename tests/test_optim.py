import numpy as np
from numcompute_stream.optim import grad
import pytest
from numcompute_stream.optim import jacobian

def test_grad_simple():
    def f(x):
        return x[0]**2 + x[1]

    g = grad(f, [2, 3])

    assert np.allclose(g, [4, 1], atol=1e-5)


def test_grad_forward_method():
    def f(x):
        return x[0]**2 + x[1]**2

    g = grad(f, [3, 4], method="forward")

    assert np.allclose(g, [6, 8], atol=1e-3)
def test_grad_negative_values():
    def f(x):
        return x[0]**2 + x[1]**2

    g = grad(f, [-2, -3])

    assert np.allclose(g, [-4, -6], atol=1e-5)


def test_grad_constant_function():
    def f(x):
        return 10.0

    g = grad(f, [1, 2, 3])

    assert np.allclose(g, [0, 0, 0], atol=1e-5)

def test_grad_empty_input():
    def f(x):
        return 0

    with pytest.raises(ValueError):
        grad(f, [])


def test_grad_invalid_method():
    def f(x):
        return x[0]

    with pytest.raises(ValueError):
        grad(f, [1, 2], method="wrong")

def test_jacobian_simple():
    def F(x):
        return np.array([x[0] + x[1], x[0] * x[1]])

    J = jacobian(F, [2, 3])

    expected = np.array([[1, 1], [3, 2]])

    assert np.allclose(J, expected, atol=1e-5)


def test_jacobian_forward():
    def F(x):
        return np.array([x[0]**2, x[1]**2])

    J = jacobian(F, [3, 4], method="forward")

    expected = np.array([[6, 0], [0, 8]])

    assert np.allclose(J, expected, atol=1e-3)

def test_jacobian_single_output():
    def F(x):
        return np.array([x[0]**2 + x[1]])

    J = jacobian(F, [2, 3])

    assert np.allclose(J, [[4, 1]], atol=1e-5)


def test_jacobian_empty_output():
    def F(x):
        return np.array([])

    import pytest
    with pytest.raises(ValueError):
        jacobian(F, [1, 2])
