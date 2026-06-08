"""
tests/test_rank.py
Unit tests for numcompute/rank.py
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'numcompute'))

from numcompute.rank import rank, percentile



# rank tests

class TestRankOrdinal(unittest.TestCase):

    def test_no_ties(self):
        """No ties — each element gets unique rank."""
        result = rank(np.array([3., 1., 4., 2.]), method='ordinal')
        np.testing.assert_array_equal(result, np.array([3., 1., 4., 2.]))

    def test_with_ties_first_occurrence_lower(self):
        """First occurrence of tied value gets lower rank."""
        result = rank(np.array([3., 1., 4., 1., 5.]), method='ordinal')
        np.testing.assert_array_equal(result, np.array([3., 1., 4., 2., 5.]))

    def test_all_equal(self):
        """All equal — ranks should be 1,2,3 by position."""
        result = rank(np.array([5., 5., 5.]), method='ordinal')
        np.testing.assert_array_equal(result, np.array([1., 2., 3.]))

    def test_single_element(self):
        """Single element always gets rank 1."""
        result = rank(np.array([42.]), method='ordinal')
        np.testing.assert_array_equal(result, np.array([1.]))

    def test_reverse_sorted(self):
        """Reverse sorted input should still rank correctly."""
        result = rank(np.array([4., 3., 2., 1.]), method='ordinal')
        np.testing.assert_array_equal(result, np.array([4., 3., 2., 1.]))


class TestRankDense(unittest.TestCase):

    def test_no_ties(self):
        """No ties — dense ranks equal ordinal ranks."""
        result = rank(np.array([3., 1., 4., 2.]), method='dense')
        np.testing.assert_array_equal(result, np.array([3., 1., 4., 2.]))

    def test_with_ties_no_gaps(self):
        """Tied elements share lowest rank, no gaps after."""
        result = rank(np.array([3., 1., 4., 1., 5.]), method='dense')
        np.testing.assert_array_equal(result, np.array([2., 1., 3., 1., 4.]))

    def test_all_equal(self):
        """All equal — everyone gets rank 1."""
        result = rank(np.array([5., 5., 5.]), method='dense')
        np.testing.assert_array_equal(result, np.array([1., 1., 1.]))

    def test_three_way_tie(self):
        """Three-way tie all get rank 1, next gets rank 2."""
        result = rank(np.array([2., 2., 2., 5.]), method='dense')
        np.testing.assert_array_equal(result, np.array([1., 1., 1., 2.]))

    def test_two_distinct_groups(self):
        """Two distinct values — lower group rank 1, higher rank 2."""
        result = rank(np.array([10., 5., 10., 5.]), method='dense')
        np.testing.assert_array_equal(result, np.array([2., 1., 2., 1.]))


class TestRankAverage(unittest.TestCase):

    def test_no_ties(self):
        """No ties — average ranks equal ordinal ranks."""
        result = rank(np.array([3., 1., 4., 2.]), method='average')
        np.testing.assert_array_equal(result, np.array([3., 1., 4., 2.]))

    def test_two_way_tie(self):
        """Two-way tie occupying positions 1,2 → average rank 1.5."""
        result = rank(np.array([3., 1., 4., 1., 5.]), method='average')
        np.testing.assert_array_equal(result, np.array([3., 1.5, 4., 1.5, 5.]))

    def test_three_way_tie(self):
        """Three-way tie occupying positions 2,3,4 → average rank 3."""
        result = rank(np.array([2., 2., 2., 1.]), method='average')
        np.testing.assert_array_equal(result, np.array([3., 3., 3., 1.]))

    def test_all_equal_three(self):
        """All equal — everyone gets mean of all ranks."""
        result = rank(np.array([5., 5., 5.]), method='average')
        np.testing.assert_array_equal(result, np.array([2., 2., 2.]))

    def test_two_groups_tied(self):
        """Two groups each tied — each group gets its mean rank."""
        result = rank(np.array([1., 1., 3., 3.]), method='average')
        np.testing.assert_array_equal(result, np.array([1.5, 1.5, 3.5, 3.5]))


class TestRankEdgeCases(unittest.TestCase):

    def test_invalid_method_raises(self):
        """Unrecognised method should raise ValueError."""
        with self.assertRaises(ValueError):
            rank(np.array([1., 2., 3.]), method='unknown')

    def test_2d_input_raises(self):
        """2D input should raise ValueError."""
        with self.assertRaises(ValueError):
            rank(np.array([[1., 2.], [3., 4.]]))

    def test_list_input_accepted(self):
        """Should accept Python lists."""
        result = rank([3., 1., 2.], method='ordinal')
        np.testing.assert_array_equal(result, np.array([3., 1., 2.]))

    def test_returns_float_dtype(self):
        """Output dtype should always be float."""
        result = rank(np.array([1., 2., 3.]), method='average')
        self.assertEqual(result.dtype, float)

    def test_negative_values(self):
        """Should handle negative values correctly."""
        result = rank(np.array([-3., -1., -2.]), method='ordinal')
        np.testing.assert_array_equal(result, np.array([1., 3., 2.]))



# percentile tests

class TestPercentileInterpolation(unittest.TestCase):

    def setUp(self):
        self.data5 = np.array([1., 2., 3., 4., 5.])
        self.data4 = np.array([1., 2., 3., 4.])

    def test_median_odd_length(self):
        """Median of odd-length array hits exact index."""
        self.assertEqual(percentile(self.data5, 50), 3.0)

    def test_q0_returns_minimum(self):
        """q=0 should return minimum."""
        self.assertEqual(percentile(self.data5, 0), 1.0)

    def test_q100_returns_maximum(self):
        """q=100 should return maximum."""
        self.assertEqual(percentile(self.data5, 100), 5.0)

    def test_linear_interpolation(self):
        """Linear interpolation between two neighbours."""
        result = percentile(self.data4, 50, interpolation='linear')
        self.assertEqual(result, 2.5)

    def test_lower_interpolation(self):
        """Lower interpolation takes the floor neighbour."""
        result = percentile(self.data4, 50, interpolation='lower')
        self.assertEqual(result, 2.0)

    def test_higher_interpolation(self):
        """Higher interpolation takes the ceiling neighbour."""
        result = percentile(self.data4, 50, interpolation='higher')
        self.assertEqual(result, 3.0)

    def test_midpoint_interpolation(self):
        """Midpoint interpolation takes mean of two neighbours."""
        result = percentile(self.data4, 50, interpolation='midpoint')
        self.assertEqual(result, 2.5)

    def test_25th_percentile(self):
        """25th percentile of [1,2,3,4,5] should be 2.0."""
        result = percentile(self.data5, 25)
        self.assertEqual(result, 2.0)

    def test_75th_percentile(self):
        """75th percentile of [1,2,3,4,5] should be 4.0."""
        result = percentile(self.data5, 75)
        self.assertEqual(result, 4.0)


class TestPercentileEdgeCases(unittest.TestCase):

    def test_nan_values_ignored(self):
        """NaN values should be stripped before computing."""
        data = np.array([1., np.nan, 3., np.nan, 5.])
        result = percentile(data, 50)
        self.assertEqual(result, 3.0)

    def test_all_nan_raises(self):
        """All-NaN array should raise ValueError."""
        with self.assertRaises(ValueError):
            percentile(np.array([np.nan, np.nan]), 50)

    def test_single_element(self):
        """Single element — all percentiles return that element."""
        result = percentile(np.array([42.]), 50)
        self.assertEqual(result, 42.0)

    def test_multiple_q_values(self):
        """Multiple q values should return array of results."""
        result = percentile(np.array([1., 2., 3., 4., 5.]),
                            np.array([0., 50., 100.]))
        np.testing.assert_array_equal(result, np.array([1., 3., 5.]))

    def test_scalar_q_returns_scalar(self):
        """Scalar q input should return Python float, not array."""
        result = percentile(np.array([1., 2., 3.]), 50)
        self.assertIsInstance(result, float)

    def test_q_out_of_range_raises(self):
        """q outside [0,100] should raise ValueError."""
        with self.assertRaises(ValueError):
            percentile(np.array([1., 2., 3.]), 150)

    def test_bad_interpolation_raises(self):
        """Unrecognised interpolation method should raise ValueError."""
        with self.assertRaises(ValueError):
            percentile(np.array([1., 2., 3.]), 50, interpolation='unknown')

    def test_2d_input_raises(self):
        """2D input should raise ValueError."""
        with self.assertRaises(ValueError):
            percentile(np.array([[1., 2.], [3., 4.]]), 50)


# Testing

if __name__ == '__main__':
    unittest.main(verbosity=2)