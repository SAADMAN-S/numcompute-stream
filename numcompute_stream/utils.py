"""
utils.py
Utilites: distances, activations, logsumexp and batching.
"""

import numpy as np

def euclidean(x, y, axis=-1):
    """Euclidean (L2) distance between vectors along the given axis.

    Parameters:
        x, y : Input arrays
        axis : Axis along which the L2 norm is computed (default last axis).

    Returns:
        ndarray: Element wise Euclidean distance.

    Raises:
        ValueError: If 'x' and 'y' cannot be broadcast.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y

    return np.sqrt((diff * diff).sum(axis=axis))


def manhattan(x, y, axis=-1):
    """Manhattan (L1) distance between vectors along the given axis.

    Parameters:
        x, y : Input arrays
        axis : Axis along which the L1 norm is computed (default last axis).

    Returns:
        ndarray: Element wise Manhattan distance.

    Raises:
        ValueError: If 'x' and 'y' cannot be broadcast.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    return np.abs(x - y).sum(axis=axis)


def cosine_distance(x, y, axis=-1, eps=1e-10):
    """Cosine distance along the given axis.

    Parameters:
        x, y : Input arrays.
        axis : Feature axis.
        eps : Small value to avoid division by zero.

    Returns:
        ndarray: Cosine distance.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dot = (x * y).sum(axis=axis)

    nx = np.sqrt((x * x).sum(axis=axis)).clip(min=eps)
    ny = np.sqrt((y * y).sum(axis=axis)).clip(min=eps)

    return 1.0 - dot / (nx * ny)


def relu(x):
    """Rectified Linear Unit: max(0, x).

    Parameters:
        x: Input arrays.

    Returns:
        ReLU.
    """
    return np.maximum(0, np.asarray(x, dtype=float))


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: x if x > 0 else alpha * x.
    
    Parameters:
        x: Input arrays.
        alpha: Address dying ReLU.

    Returns:
        ReLU.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * x)


def sigmoid(x):
    """Sigmoid activation: 1 / (1 + exp(-x)).
    
    Parameters:
        x: Input arrays.
    
    Returns:
        ndarray: sigmoid.
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    """Hyperbolic tangent.
    
    Parameters:
        x: Input arrays.

    Returns:
        tanh.
    """
    x = np.asarray(x, dtype=float)
    return np.tanh(x)

def softmax(x, axis=-1):
    """Softmax function.

    Parameters:
        x: Input ndarray
        axis: Axis along which softmax is computed (default last axis).

    Returns:
        ndarray: Softmax probabilities.
    """
    x = np.asarray(x, dtype=float)
    lse = logsumexp(x, axis=axis, keepdims=True)

    return np.exp(x - lse)


def logsumexp(x, axis=None, keepdims=False):
    """Stable computation of log(sum(exp(x), axis)).

    Parameters:
        x: Input values.
        axis: Axis over which to sum.
        keepdims: If True, the reduced axes are kept with length 1.

    Returns:
        ndarray: logsumexp result.
    """

    x = np.asarray(x, dtype=float)

    if axis is None:
        x_max = x.max()
        return x_max + np.log(np.sum(np.exp(x - x_max)))

    else:
        x_max = x.max(axis=axis, keepdims=True)
        return (x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))).squeeze(axis=None if keepdims else axis)


def batch(*arrays, batch_size=32, shuffle=True, seed=None):
    """Yield batches from arrays.

    Parameters:
        *arrays: One or more arrays to iterate over.
        batch_size: Number of samples per batch.
        shuffle: If True, shuffle the data at the start of each epoch.
        seed: Seed for reproducible shuffling.

    Yields: Slices of the input arrays of length batch size.

    Raises:
        ValueError: If no arrays are provided or their first dimensions differ.
    """

    if not arrays:
        raise ValueError("At least one array must be provided.")

    n = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != n:
            raise ValueError("All arrays must have the same first dimension.")

    indices = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield tuple(arr[batch_idx] for arr in arrays)


