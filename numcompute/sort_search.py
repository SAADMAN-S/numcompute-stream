"""
sort_search.py

Sorting, top-k selection, and binary search utilities for NumCompute.

Author: Saadman Sakib
"""

import numpy as np

#Sorting an unordered array - returning an ordered array as output

def stable_sort(arr, axis=-1):

    """
    Stability during sorting means same values in the array do not disrupt their original relative order.
    This matters in multi-pass pipelines where you sort by one key first before another,
     — an unstable sort would disrupt the first sort's result.

    - Parameters:
    arr : array of numbers - any shape.
    axis : int, default = -1 which is the last axis.

    - Returns:
    sorted_arr : np.ndarray, Sorted copy of the original array. 
    
    - Raises:
    ValueError: If axis is out of range for the array's dimension count - ndim.
    

    - Complexity:
    Time  : O(n log n) — mergesort, same in all cases - best and worst.
    Space : O(n) — mergesort requires additional space.

    """
    
    arr = np.asarray(arr)

    if arr.ndim == 0 :
        return arr.copy()

    if not (-arr.ndim <= axis < arr.ndim):
        raise ValueError(
            f"Axis {axis} is outside the bounds for the array of dimensions {arr.ndim}."
        )
    return np.sort(arr,axis = axis, kind = "stable")


def multi_key_sort(arr, keys, ascending=None):
    """
    Sort a 2-D array by multiple columns, with left-to-right key priority.
    
    - Parameters:
    arr : array_like, shape (n, m), 2-D array where each row is one record.
    keys : list of int, Column indices in priority order. keys[0] = primary sort key.
    ascending : list of bool or None, optional
        Per-key direction. True = ascending, False = descending.
        Ascending by-default if None.
    
    - Returns :
    sorted_arr : np.ndarray, shape (n, m)

    - Raises ValueError:
        If ``arr`` is not 2-D, ``keys`` is empty , any column index is out of range, ``ascending`` has wrong length

    - Complexity:
    Time  : O(n log n) — np.lexsort performs one pass per key.
    Space : O(n)       — stores a row-index array of length n.
    
    """

    arr = np.asarray(arr,dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"The arr must be 2-D, but the sample input is {arr.shape}.")
    
    n,m = arr.shape

    if not keys:
        raise ValueError("Keys must be a non-empty list of column indices.")
    
    for k in keys:
        if not (0 <= k < m):
            raise ValueError(f"Column index {k} is out of range for array having {m} columns.")
        
    if ascending is None:
        ascending = [True]*len(keys)
    
    if len(ascending) != len(keys):
        raise ValueError("The length of ascending must be equal to the length of keys.")

    
    #Extracting the key columns using fancy indexing
    ascending = np.array(ascending,dtype=float)
    multiplier = np.where(ascending, 1.0, -1.0)

    #broadcast multipliers across all rows at once
    cols_adjusted = arr[:,keys] * multiplier

    #Transpose cols_adjusted then reversed [::-1] so keys[0] (the primary key) becomes the last row, having the highest priority
    lex_input = cols_adjusted.T[::-1]

    order = np.lexsort(lex_input)
    return arr[order]


def topk(arr, k, largest=True, return_indices=True):
    """
    partial sorting which uses introspect - a combination of Quickselect and mergesort fallback

    - Parameters: 
        values : array_like, shape (n,) 1-D numeric input array.
    k : int
        Number of elements to retrieve. Must satisfy 1 <= k <= n.
   
    - Returns:
    top_values : np.ndarray, shape (k,)
        Selected values, sorted descending (largest=True) or ascending (largest=False).
    top_indices : np.ndarray, shape (k,)
        Indices into original values and only when return_indices=True.

    

    - Raises:
    ValueError if values is not 1d, or k is outside [1, n].


    - Complexity:
    Time  : O(n) average for argpartition + O(k log k) to sort k results.
    Space : O(k).

    """
    arr = np.asarray(arr, dtype=float) 

    if arr.ndim != 1:
        raise ValueError("The arr must be 1D.")

    n = len(arr)

    if not (1 <= k <=n):
        raise ValueError("The value of k must be in between 1 and n.")
    
    part_index = []
    
    if largest:
        part_index = np.argpartition(arr, -k)[-k:]
        sorted_order = np.argsort(arr[part_index])[::-1]

    else:
        part_index = np.argpartition(arr, k)[:k]
        sorted_order = np.argsort(arr[part_index])

    
    top_indices = part_index[sorted_order]
    top_elements = arr[top_indices]

    if return_indices:
        return top_elements, top_indices
    
    return top_elements


def quickselect(arr, k):
    """
    Find the k-th smallest element (0-indexed) using Quickselect.

    Educational implementation — shows the algorithm .
    
    - Parameters:
    arr : array_like, shape (n,)
        1-D numeric input array.
    k : int
        0-based index in sorted order.
        k=0 → minimum, k=n-1 → maximum.

    
    - Returns:
    element : float
        The k-th smallest value in ``arr``.


    - Raises:
    ValueError
        If ``arr`` is not 1-D or k is outside the range 0, n-1.

    - Complexity:
    Time  : O(n) average, O(n²) worst case (bad pivot).
    Space : O(n) 


    """
    
    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"The arr must be 1D.")

    n = len(arr)

    if not (0 <= k < n):
        raise ValueError(f"The value of k must be within the range 0 to n-1.")
    
    data = arr.copy()

    def _partition(low, high):
        """
        Initially,
        Pivot = data[high]

        Thus, moving pivot to its correct final position.
        The final pivot index is returned.
        """

        pivot = data[high]
        i = low

        for j in range(low, high):
            if data[j] <= pivot:
                data[i], data[j] = data[j], data[i]
                i += 1

        data[i], data[high] = data[high], data[i]
        return i

    
    low, high = 0, n-1

    while low < high:
            
        pivot_position = _partition(low, high)

        if pivot_position == k:
            break
        elif pivot_position < k:
            low = pivot_position + 1
        else:
            high = pivot_position - 1
    
    return data[k]

#Searching an array to find one particular value - returning a single value from an array
def binary_search(arr, x):

    """
    Search for x in a sorted 1d array using binary search.

    Wraps np.searchsorted (O(log n)) and additionally returns
    whether x actually exists in the array.


    - Parameters:
    sorted_array : array_like, shape (n,)
        1-D array sorted in ascending order.
    x : scalar

    - Returns: 
    index : int
        Insertion index — where x would be inserted to keep the
        array sorted.
    found : bool
        True if x exists in sorted_array, False otherwise.
    

    - Raises:
    ValueError - If sorted_array is not 1d.

    
    - Complexity:
    Time  : O(log n)
    Space : O(1).

    """

    sorted_array = np.asarray(arr)

    if sorted_array.ndim != 1:
        raise ValueError(
            f"The sorted array must be 1D."
        )
        
    index = int(np.searchsorted(sorted_array, x, side="left"))

    found = (
        index < len(sorted_array)
        and bool(sorted_array[index] == x)
        )
    return index, found


     