"""
io.py
Read CSV file and return ndarray

Author: Risat Rahaman
"""

import numpy as np

def load_csv(filepath, delimiter=',', fill_missing_value='nan'):
    """Load a CSV file and return the header and data.

    The file is read with `numpy.genfromtxt` as a string array. Empty strings
    in the data body are replaced by `fill_missing_value`. An attempt is made
    to cast the data to float; if that fails, the string data is returned.

    Parameters:
        filepath (str): Path to the CSV file.
        delimiter (str): Field delimiter (default ',').
        fill_missing_value (str): Value used to replace empty strings (default 'nan').

    Returns:
        tuple: A tuple (header, data). header is a 1D ndarray of column names,
               data is a 2D ndarray of values. The data array is of float type
               if conversion succeeds; otherwise it remains a string array.
    """
    raw = np.genfromtxt(filepath, delimiter=delimiter, dtype='U100')

    if raw.size == 0:
        return np.array([]), np.array([])

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    header = raw[0]
    data = raw[1:]

    data[data == ''] = fill_missing_value

    try:
        return header, data.astype(float)
    except ValueError:
        return header, data


def get_column(column_name, data, header):
    """Extract a column from a 2D array using a header array.

    Parameters:
        column_name (str): Name of the column to extract.
        data (ndarray): 2D array from which the column is taken.
        header (ndarray): 1D array of column names corresponding to the columns
                          of 'data'.

    Returns:
        ndarray: 1D array of values from the requested column.

    Raises:
        IndexError: If 'column_name' is not found in 'header'.
    """
    col_index = np.where(header == column_name)[0][0]
    return data[:, col_index]