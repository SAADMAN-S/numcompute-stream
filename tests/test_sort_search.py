"""
tests/test_sort_search.py
=========================
Unit tests for numcompute/sort_search.py
Author: Saadman Sakib
"""

import sys
import os
import unittest
import numpy as np

# Make Python find numcompute/ from the tests/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'numcompute'))

from numcompute.sort_search import stable_sort, multi_key_sort, topk, quickselect, binary_search



# stable_sort tests


class TestStableSort(unittest.TestCase):

    def test_basic_1d_ascending(self):
        """Basic 1D sort should return ascending order."""
        result = stable_sort(np.array([3, 1, 2, 1]))
        expected = np.array([1, 1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_already_sorted(self):
        """Already sorted array should be unchanged."""
        result = stable_sort(np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

    def test_all_equal_values(self):
        """All-equal array should return same values."""
        result = stable_sort(np.array([5, 5, 5, 5]))
        np.testing.assert_array_equal(result, np.array([5, 5, 5, 5]))

    def test_2d_sort_axis0(self):
        """2D array sorted along axis=0 should sort each column."""
        result = stable_sort(np.array([[3, 1], [2, 4]]), axis=0)
        expected = np.array([[2, 1], [3, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_2d_sort_axis1(self):
        """2D array sorted along axis=1 should sort each row."""
        result = stable_sort(np.array([[3, 1], [4, 2]]), axis=1)
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_single_element(self):
        """Single element array should return same element."""
        result = stable_sort(np.array([42]))
        np.testing.assert_array_equal(result, np.array([42]))

    def test_list_input_accepted(self):
        """Should accept Python lists, not just numpy arrays."""
        result = stable_sort([3, 1, 2])
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_original_not_mutated(self):
        """Original array must not be modified."""
        arr = np.array([3, 1, 2])
        original = arr.copy()
        stable_sort(arr)
        np.testing.assert_array_equal(arr, original)

    def test_invalid_axis_raises(self):
        """Out-of-range axis should raise ValueError."""
        with self.assertRaises(ValueError):
            stable_sort(np.array([1, 2, 3]), axis=5)

    def test_stability_equal_elements(self):
        """Stable sort must preserve original order for equal elements."""
        # Use structured array to verify stability
        arr = np.array([2, 1, 2, 1])
        result = stable_sort(arr)
        # All 1s before 2s
        self.assertTrue(result[0] == 1 and result[1] == 1)



# multi_key_sort tests


class TestMultiKeySort(unittest.TestCase):

    def test_single_key_ascending(self):
        """Sort by one column ascending."""
        data = np.array([[3., 2.], [1., 4.], [2., 1.]])
        result = multi_key_sort(data, keys=[0])
        self.assertEqual(result[0, 0], 1.0)
        self.assertEqual(result[2, 0], 3.0)

    def test_two_keys_tie_breaking(self):
        """Primary key ties broken by secondary key."""
        data = np.array([[3., 2.], [1., 4.], [1., 2.]])
        result = multi_key_sort(data, keys=[0, 1])
        expected = np.array([[1., 2.], [1., 4.], [3., 2.]])
        np.testing.assert_array_equal(result, expected)

    def test_single_key_descending(self):
        """Sort by one column descending."""
        data = np.array([[1., 5.], [3., 2.], [2., 8.]])
        result = multi_key_sort(data, keys=[0], ascending=[False])
        self.assertEqual(result[0, 0], 3.0)

    def test_mixed_ascending_descending(self):
        """First key ascending, second key descending."""
        data = np.array([[1., 3.], [1., 1.], [2., 2.]])
        result = multi_key_sort(data, keys=[0, 1], ascending=[True, False])
        expected = np.array([[1., 3.], [1., 1.], [2., 2.]])
        np.testing.assert_array_equal(result, expected)

    def test_original_not_mutated(self):
        """Original array must not be modified."""
        data = np.array([[3., 2.], [1., 4.]])
        original = data.copy()
        multi_key_sort(data, keys=[0])
        np.testing.assert_array_equal(data, original)

    def test_empty_keys_raises(self):
        """Empty keys list should raise ValueError."""
        with self.assertRaises(ValueError):
            multi_key_sort(np.array([[1., 2.]]), keys=[])

    def test_invalid_column_index_raises(self):
        """Out-of-range column index should raise ValueError."""
        with self.assertRaises(ValueError):
            multi_key_sort(np.array([[1., 2.]]), keys=[99])

    def test_1d_input_raises(self):
        """1D input should raise ValueError."""
        with self.assertRaises(ValueError):
            multi_key_sort(np.array([1., 2., 3.]), keys=[0])

    def test_ascending_length_mismatch_raises(self):
        """ascending list length != keys length should raise ValueError."""
        with self.assertRaises(ValueError):
            multi_key_sort(
                np.array([[1., 2.], [3., 4.]]),
                keys=[0, 1],
                ascending=[True]
            )



# topk tests


class TestTopK(unittest.TestCase):

    def setUp(self):
        self.v = np.array([3., 1., 4., 1., 5., 9., 2., 6.])

    def test_top3_largest_values(self):
        """Top 3 largest should be 9, 6, 5."""
        vals, _ = topk(self.v, k=3, largest=True)
        np.testing.assert_array_equal(np.sort(vals)[::-1], np.array([9., 6., 5.]))

    def test_top3_smallest_values(self):
        """Top 3 smallest should be 1, 1, 2."""
        vals, _ = topk(self.v, k=3, largest=False)
        np.testing.assert_array_equal(np.sort(vals), np.array([1., 1., 2.]))

    def test_return_indices_true(self):
        """return_indices=True should return tuple of (values, indices)."""
        result = topk(self.v, k=3, return_indices=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_return_indices_false(self):
        """return_indices=False should return only values array."""
        result = topk(self.v, k=3, return_indices=False)
        self.assertIsInstance(result, np.ndarray)

    def test_indices_point_to_correct_values(self):
        """Returned indices must point to correct values in original array."""
        vals, idxs = topk(self.v, k=3, largest=True)
        np.testing.assert_array_equal(vals, self.v[idxs])

    def test_k_equals_1(self):
        """k=1 largest should return single maximum."""
        vals, _ = topk(self.v, k=1, largest=True)
        self.assertEqual(vals[0], 9.0)

    def test_k_equals_n(self):
        """k=n should return all elements."""
        vals, _ = topk(self.v, k=len(self.v), largest=True)
        self.assertEqual(len(vals), len(self.v))

    def test_k_zero_raises(self):
        """k=0 should raise ValueError."""
        with self.assertRaises(ValueError):
            topk(self.v, k=0)

    def test_k_exceeds_n_raises(self):
        """k > n should raise ValueError."""
        with self.assertRaises(ValueError):
            topk(self.v, k=999)

    def test_2d_input_raises(self):
        """2D input should raise ValueError."""
        with self.assertRaises(ValueError):
            topk(np.array([[1., 2.], [3., 4.]]), k=2)



# quickselect tests


class TestQuickselect(unittest.TestCase):

    def setUp(self):
        self.arr = np.array([3., 1., 4., 1., 5., 9., 2., 6.])

    def test_minimum_k0(self):
        """k=0 should return minimum element."""
        result = quickselect(self.arr, k=0)
        self.assertEqual(result, 1.0)

    def test_maximum_k_last(self):
        """k=n-1 should return maximum element."""
        result = quickselect(self.arr, k=7)
        self.assertEqual(result, 9.0)

    def test_middle_element(self):
        """k=3 should match sorted array at index 3."""
        result = quickselect(self.arr, k=3)
        expected = float(np.sort(self.arr)[3])
        self.assertEqual(result, expected)

    def test_all_equal_values(self):
        """All-equal array should return that value for any k."""
        result = quickselect(np.array([7., 7., 7., 7.]), k=2)
        self.assertEqual(result, 7.0)

    def test_single_element(self):
        """Single element array with k=0 should return that element."""
        result = quickselect(np.array([42.]), k=0)
        self.assertEqual(result, 42.0)

    def test_original_not_mutated(self):
        """Original array must not be modified by quickselect."""
        original = self.arr.copy()
        quickselect(self.arr, k=3)
        np.testing.assert_array_equal(self.arr, original)

    def test_k_negative_raises(self):
        """Negative k should raise ValueError."""
        with self.assertRaises(ValueError):
            quickselect(self.arr, k=-1)

    def test_k_out_of_range_raises(self):
        """k >= n should raise ValueError."""
        with self.assertRaises(ValueError):
            quickselect(self.arr, k=99)

    def test_2d_input_raises(self):
        """2D input should raise ValueError."""
        with self.assertRaises(ValueError):
            quickselect(np.array([[1., 2.], [3., 4.]]), k=0)

    def test_duplicate_values(self):
        """Should correctly find k-th element when duplicates exist."""
        arr = np.array([5., 3., 3., 1., 3.])
        result = quickselect(arr, k=2)
        expected = float(np.sort(arr)[2])
        self.assertEqual(result, expected)



# binary_search tests


class TestBinarySearch(unittest.TestCase):

    def setUp(self):
        self.sarr = np.array([1., 3., 5., 7., 9.])

    def test_element_found_middle(self):
        """Element in middle should be found at correct index."""
        idx, found = binary_search(self.sarr, 5.)
        self.assertEqual(idx, 2)
        self.assertTrue(found)

    def test_element_found_first(self):
        """First element should be found at index 0."""
        idx, found = binary_search(self.sarr, 1.)
        self.assertEqual(idx, 0)
        self.assertTrue(found)

    def test_element_found_last(self):
        """Last element should be found at correct index."""
        idx, found = binary_search(self.sarr, 9.)
        self.assertEqual(idx, 4)
        self.assertTrue(found)

    def test_element_not_found_middle(self):
        """Missing element should return correct insertion index."""
        idx, found = binary_search(self.sarr, 4.)
        self.assertEqual(idx, 2)
        self.assertFalse(found)

    def test_element_smaller_than_all(self):
        """Element smaller than all should give insertion index 0."""
        idx, found = binary_search(self.sarr, 0.)
        self.assertEqual(idx, 0)
        self.assertFalse(found)

    def test_element_larger_than_all(self):
        """Element larger than all should give insertion index n."""
        idx, found = binary_search(self.sarr, 10.)
        self.assertEqual(idx, 5)
        self.assertFalse(found)

    def test_empty_array(self):
        """Empty array should return index 0 and found=False."""
        idx, found = binary_search(np.array([]), 5.)
        self.assertEqual(idx, 0)
        self.assertFalse(found)

    def test_single_element_found(self):
        """Single element array, element present."""
        idx, found = binary_search(np.array([7.]), 7.)
        self.assertEqual(idx, 0)
        self.assertTrue(found)

    def test_single_element_not_found(self):
        """Single element array, element absent."""
        idx, found = binary_search(np.array([7.]), 3.)
        self.assertEqual(idx, 0)
        self.assertFalse(found)

    def test_2d_input_raises(self):
        """2D input should raise ValueError."""
        with self.assertRaises(ValueError):
            binary_search(np.array([[1., 2.], [3., 4.]]), 2.)


# testing

if __name__ == '__main__':
    unittest.main(verbosity=2)