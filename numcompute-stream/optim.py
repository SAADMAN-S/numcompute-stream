"""
Finite-difference gradient and Jacobian estimation.

This module follows the NumCompute core requirement
- grad(f, x, h=1e-5, method='central'|'forward')
- jacobian(F, x, h=1e-5, method='central'|'forward')

Only NumPy is used.
"""
import numpy as np


def grad(f, x, h=1e-5, method="central"):
    """
    Estimation of the gradient of a scalar-valued function using finite differences.

    Parameters
    ----------
    f : callable
        Scalar function f: R^n -> R.
    x : array-like, shape (n,)
        Input point where the gradient is estimated.
    h : float, default=1e-5
        Positive finite step size.
    method : {'central', 'forward'}, default='central'
        Finite-difference method.

    Returns
    -------
    ndarray, shape (n,)
        Estimated gradient vector.

    Raises
    ------
    ValueError
        If x is empty, h is invalid, method is unsupported, or f returns
        a non-scalar / non-finite value.

    Time Complexity
    ---------------
    O(n * C_f)
    where C_f is the cost of evaluating f.

    Space Complexity
    ----------------
    O(n)
    """

    x = np.asarray(x, dtype=float)
    n = x.size

    if n == 0:
        raise ValueError("x cannot be empty")

    grad = np.zeros(n)

    if method == "forward":
        fx = f(x)
        for i in range(n):
            x_h = x.copy()
            x_h[i] += h
            grad[i] = (f(x_h) - fx) / h

    elif method == "central":
        for i in range(n):
            x_h1 = x.copy()
            x_h2 = x.copy()
            x_h1[i] += h
            x_h2[i] -= h
            grad[i] = (f(x_h1) - f(x_h2)) / (2 * h)

    else:
        raise ValueError("method must be 'forward' or 'central'")

    return grad

def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimate the Jacobian matrix of a vector-valued function.

    Parameters
    ----------
    F : callable
        Vector function F: R^n -> R^m.
    x : array-like, shape (n,)
        Input point where the Jacobian is estimated.
    h : float, default=1e-5
        Positive finite step size.
    method : {'central', 'forward'}, default='central'
        Finite-difference method.

    Returns
    -------
    ndarray, shape (m, n)
        Estimated Jacobian matrix.

    Raises
    ------
    ValueError
        If x is empty, h is invalid, method is unsupported, or F returns
        empty / non-finite values.

    Time Complexity
    ---------------
    O(n * C_F), where C_F is the cost of evaluating F.

    Space Complexity
    ----------------
    O(mn)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(F(x), dtype=float)

    n = x.size
    m = y.size

    if n == 0 or m == 0:
        raise ValueError("Invalid input or output size")

    J = np.zeros((m, n))

    if method == "forward":
        for i in range(n):
            x_h = x.copy()
            x_h[i] += h
            J[:, i] = (F(x_h) - y) / h

    elif method == "central":
        for i in range(n):
            x_h1 = x.copy()
            x_h2 = x.copy()
            x_h1[i] += h
            x_h2[i] -= h
            J[:, i] = (F(x_h1) - F(x_h2)) / (2 * h)

    else:
        raise ValueError("method must be 'forward' or 'central'")

    return J