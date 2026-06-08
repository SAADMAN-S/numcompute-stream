"""
rank.py

Ranking and percentile utilities for NumCompute.
Author: Saadman Sakib
"""

import numpy as np

def rank(data, method = "average"):
    """
    Dealing with ties - assigning ranks to elements in the 1d array.

    - Parameters:
    data : array, shape (n,)
        1D numeric input array. method : {'average', 'dense', 'ordinal'}
        How to handle ties:
        'average' — tied elements share the mean of their ranks - divided by 2.
        'dense'   — tied elements share the lowest rank, no gaps.
        'ordinal' — each element gets a unique rank starting from 1 order.

    - Returns:
    ranks : np.ndarray, shape (n,), dtype float. Ranks start at 1.

    - Raises:
    ValueError if data is not 1D, or method is not recognised.

    - Complexity:
    Time:O(nlog n) due to argsort.
    Space:O(n).
    """
    
    data = np.asarray(data, dtype = float)

    if data.ndim != 1:
        raise ValueError(f"Data must be 1D but given input has shape {data.shape}.")
    
    valid_methods = {'average', 'dense','ordinal'}

    if method not in valid_methods:
        raise ValueError(
            f"Method must be one of {valid_methods}, got '{method}'."
        )
    
    n = len(data)

    sorter = np.argsort(data, kind = 'stable')

    if method == 'ordinal':
        ranks = np.empty(n, dtype = float)
        ranks[sorter] = np.arange(1, n+1, dtype = float)
    
    elif method == 'dense':
        _,unique_inv = np.unique(data, return_inverse = True)
        ranks = (unique_inv+1).astype(float)
    
    else:
        ranks = np.empty(n, dtype=float)
        ranks[sorter] = np.arange(1, n + 1, dtype=float)

        # detect where values change in sorted order
        sorted_data = data[sorter]
        obs = np.concatenate(([True], sorted_data[1:] != sorted_data[:-1]))

        # cumsum gives group label in each SORTED position
        dense = obs.cumsum()

        group_labels = np.empty(n, dtype=int)
        group_labels[sorter] = dense

        # sum and count ranks per group
        group_sum   = np.bincount(group_labels - 1, weights=ranks)
        group_count = np.bincount(group_labels - 1)
        avg_rank    = group_sum / group_count

        ranks = avg_rank[group_labels - 1]

    return ranks


def percentile(data, q, interpolation='linear'):
    """
    Computing the qth percentile in a 1d array.

    - Parameters:
    data : array, shape (n,). Ignore null values
    q : float or array_like of float, values are computed percentiles within the range 0 to 100.
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, optional
        Methods when target percentile falls between two points:
        'linear'   — weighted average of the two neighbouring points.
        'lower'    — take the lower neighbouring point.
        'higher'   — take the higher neighbouring point.
        'midpoint' — arithmetic mean of the two neighbouring point.

    - Returns:
    result : float or np.ndarray. Output is scalar if q is scalar, array otherwise.

    - Raises:
    ValueError
        If data is not 1d array, any q value falls outside the range 0 to 100, interpolation
        is not recognised, or data has no valid values after NaN removal.

    - Complexity:
    Time  : O(nlog n) due to sort.
    Space : O(n).
    """

    data = np.asarray(data, dtype = float)

    if data.ndim != 1:
        raise ValueError(f"Data must be 1D, but the input has shape {data.shape}")

    valid_interp = {'linear','lower','higher','midpoint'}

    if interpolation not in valid_interp:
        raise ValueError(f"Interpolation must be one of {valid_interp}, got '{interpolation}'")
    
    q = np.asarray(q, dtype = float)
    scalar_input = q.ndim==0
    q = np.atleast_1d(q)

    if np.any((q<0) | (q>100)):
        raise ValueError("All q values must be within the range 0 to 100")
    

    #strip NaNs before computing

    data = data[~np.isnan(data)]
    n = len(data)

    if n == 0:
        raise ValueError("Data has no valid values after removal of all the NaN values.")


    sorted_data = np.sort(data, kind = "stable")


    #convert percentile [0, 100] to float index [0, n-1]
    virtual_index = (q /100.0) * (n-1)

    low = np.floor(virtual_index).astype(int)
    high = np.ceil(virtual_index).astype(int)

    #clip protects against floatng point edge cases
    low = np.clip(low, 0, n-1)
    high = np.clip(high, 0,n-1)

    low_val = sorted_data[low]
    high_val = sorted_data[high]

    if interpolation == "lower":
        result = low_val
    elif interpolation == "higher":
        result = high_val
    elif interpolation == "midpoint":
        result = (low_val + high_val)/2.0
    else:
        fraction = virtual_index - low
        result = low_val + fraction* (high_val - low_val)
    
    return float(result[0]) if scalar_input else result