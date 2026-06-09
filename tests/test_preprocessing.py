"""
test_preprocessing.py
Unit tests for preprocessing: StandardScaler, MinMaxScaler, OneHotEncoder
"""

import unittest
import numpy as np
import tempfile
import os, sys

sys.path.append("../numcompute")

from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer

class TestStandardScaler(unittest.TestCase):
    """Tests for StandardScaler."""

    def setUp(self):
        """Set up a simple 3x2 dataset for testing."""
        self.X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_fit(self):
        """Check that mean and std are correctly computed."""
        scaler = StandardScaler()
        scaler.fit(self.X)

        np.testing.assert_array_almost_equal(scaler.mean, [3.0, 4.0])
        np.testing.assert_array_almost_equal(scaler.std, [1.63299316, 1.63299316])

    def test_transform(self):
        """Verify that transformed data has zero mean and unit variance."""
        scaler = StandardScaler()
        scaler.fit(self.X)

        X_scaled = scaler.transform(self.X)
        mean = np.mean(X_scaled, axis=0)
        std = np.std(X_scaled, axis=0)

        np.testing.assert_array_almost_equal(mean, [0.0, 0.0])
        np.testing.assert_array_almost_equal(std, [1.0, 1.0])

    def test_fit_transform(self):
        """fit_transform should return scaled data with zero mean."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        self.assertEqual(X_scaled.shape, self.X.shape)

        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), [0.0, 0.0])

    def test_constant_feature(self):
        """A constant feature should have std=1 and be scaled to zero."""
        X = np.array([[5, 1], [5, 2], [5, 3]], dtype=float)
        scaler = StandardScaler()
        scaler.fit(X)

        self.assertEqual(scaler.std[0], 1.0)

        X_scaled = scaler.transform(X)
        np.testing.assert_array_almost_equal(X_scaled[:, 0], [0.0, 0.0, 0.0])

    def test_with_nan(self):
        """NaN values should be ignored in fit statistics and propagate in transform."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        scaler = StandardScaler()
        scaler.fit(X)

        self.assertFalse(np.isnan(scaler.mean[1]))

        X_scaled = scaler.transform(X)
        self.assertTrue(np.isnan(X_scaled[0, 1]))
        self.assertFalse(np.isnan(X_scaled[0, 0]))

    def test_transform_before_fit(self):
        """Calling transform before fit should raise ValueError."""
        scaler = StandardScaler()

        with self.assertRaises(ValueError):
            scaler.transform(self.X)

    def test_1d_input(self):
        """The scaler should work on 1D arrays."""
        X_1d = np.array([1, 2, 3], dtype=float)
        scaler = StandardScaler()
        scaler.fit(X_1d)

        self.assertAlmostEqual(scaler.mean, 2.0)

        X_scaled = scaler.transform(X_1d)
        np.testing.assert_array_almost_equal(np.mean(X_scaled), 0.0)


class TestMinMaxScaler(unittest.TestCase):
    """Tests for MinMaxScaler."""

    def setUp(self):
        """Set up a simple 3x2 dataset for testing."""
        self.X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_fit_default_range(self):
        """Fit should correctly compute min and max per feature (default range)."""
        scaler = MinMaxScaler()
        scaler.fit(self.X)

        np.testing.assert_array_equal(scaler.min, [1.0, 2.0])
        np.testing.assert_array_equal(scaler.max, [5.0, 6.0])

    def test_transform_default(self):
        """Default transform scales data to [0, 1]."""
        scaler = MinMaxScaler()
        scaler.fit(self.X)

        X_scaled = scaler.transform(self.X)
        expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

        np.testing.assert_array_almost_equal(X_scaled, expected)

    def test_custom_range(self):
        """Custom feature_range (-1, 1) should be respected."""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(self.X)

        X_scaled = scaler.transform(self.X)
        expected = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])

        np.testing.assert_array_almost_equal(X_scaled, expected)

    def test_constant_feature(self):
        """Constant features are handled so that max = min + 1 and they map to 0."""
        X = np.array([[5, 1], [5, 2], [5, 3]], dtype=float)
        scaler = MinMaxScaler()
        scaler.fit(X)

        self.assertEqual(scaler.max[0], scaler.min[0] + 1.0)

        X_scaled = scaler.transform(X)
        np.testing.assert_array_almost_equal(X_scaled[:, 0], [0.0, 0.0, 0.0])

        self.assertAlmostEqual(X_scaled[0, 1], 0.0)
        self.assertAlmostEqual(X_scaled[2, 1], 1.0)

    def test_with_nan(self):
        """NaN values are ignored during fit but propagate through transform."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        scaler = MinMaxScaler()
        scaler.fit(X)

        self.assertFalse(np.isnan(scaler.min[1]))

        X_scaled = scaler.transform(X)

        self.assertTrue(np.isnan(X_scaled[0, 1]))
        self.assertFalse(np.isnan(X_scaled[0, 0]))

    def test_transform_before_fit(self):
        """Calling transform before fit should raise ValueError."""
        scaler = MinMaxScaler()

        with self.assertRaises(ValueError):
            scaler.transform(self.X)

    def test_fit_transform(self):
        """fit_transform should yield values in [0, 1]."""
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X)

        self.assertEqual(X_scaled.shape, self.X.shape)
        self.assertAlmostEqual(X_scaled.min(), 0.0)
        self.assertAlmostEqual(X_scaled.max(), 1.0)


class TestOneHotEncoder(unittest.TestCase):
    """Tests for OneHotEncoder."""

    def test_fit_1d(self):
        """Fit on a 1D array should learn the correct categories."""
        X = np.array(['a', 'b', 'a', 'c'])
        encoder = OneHotEncoder()
        encoder.fit(X)
        expected_categories = [np.array(['a', 'b', 'c'])]

        self.assertEqual(len(encoder.categories), 1)
        np.testing.assert_array_equal(encoder.categories[0], expected_categories[0])

    def test_transform_1d(self):
        """Transform a 1D array into the expected one-hot matrix."""
        X = np.array(['a', 'b', 'a', 'c'])
        encoder = OneHotEncoder()
        encoder.fit(X)
        X_enc = encoder.transform(X)

        expected = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])

        np.testing.assert_array_equal(X_enc, expected)

    def test_fit_2d(self):
        """Fit on a 2D array should learn categories per column."""
        X = np.array([['cat', 'red'], ['dog', 'blue'], ['cat', 'green']])
        encoder = OneHotEncoder()
        encoder.fit(X)

        self.assertEqual(len(encoder.categories), 2)
        np.testing.assert_array_equal(encoder.categories[0], np.array(['cat', 'dog']))
        np.testing.assert_array_equal(encoder.categories[1], np.array(['blue', 'green', 'red']))

    def test_transform_2d(self):
        """Transform a 2D array into the correct one-hot matrix."""
        X = np.array([['cat', 'red'], ['dog', 'blue'], ['cat', 'green']])
        encoder = OneHotEncoder()
        encoder.fit(X)
        X_enc = encoder.transform(X)

        expected_col0 = np.array([[1, 0], [0, 1], [1, 0]])
        expected_col1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        expected = np.hstack([expected_col0, expected_col1])

        np.testing.assert_array_equal(X_enc, expected)

    def test_fit_transform(self):
        """fit_transform should return the correct one-hot encoding."""
        X = np.array(['x', 'y', 'x'])
        encoder = OneHotEncoder()
        X_enc = encoder.fit_transform(X)

        expected = np.array([[1, 0], [0, 1], [1, 0]])
        np.testing.assert_array_equal(X_enc, expected)

    def test_unseen_categories_in_transform(self):
        """Unseen categories should yield all‑zero rows."""
        X_fit = np.array(['a', 'b'])
        encoder = OneHotEncoder()
        encoder.fit(X_fit)

        X_transform = np.array(['a', 'c'])
        X_enc = encoder.transform(X_transform)
        expected = np.array([[1, 0], [0, 0]])

        np.testing.assert_array_equal(X_enc, expected)

    def test_transform_before_fit(self):
        """Calling transform before fit should raise ValueError."""
        encoder = OneHotEncoder()

        with self.assertRaises(ValueError):
            encoder.transform(np.array(['a']))

    def test_1d_reshaping(self):
        """1D input should be reshaped correctly and produce a 2D one hot output."""
        X = np.array(['x', 'y', 'z'])
        encoder = OneHotEncoder()
        encoder.fit(X)

        X_enc = encoder.transform(X)
        self.assertEqual(X_enc.shape, (3, 3))

class TestSimpleImputer(unittest.TestCase):
    """Tests for SimpleImputer."""

    def test_fit_returns_self(self):
        """fit() should return the imputer instance itself."""
        imputer = SimpleImputer()
        X = np.array([[1, 2], [3, 4]])

        self.assertIs(imputer.fit(X), imputer)

    def test_transform_without_fit(self):
        """transform should work even if fit hasn't been explicitly called."""
        imputer = SimpleImputer(fill_value=0)
        X = np.array([[1.0, np.nan], [np.nan, 3.0]])
       
        X_imputed = imputer.transform(X)
        expected = np.array([[1.0, 0.0], [0.0, 3.0]])

        np.testing.assert_array_equal(X_imputed, expected)

    def test_default_fill_nan(self):
        """Default: replace np.nan with 0."""
        X = np.array([[1.0, np.nan], [np.nan, 3.0]])
        imputer = SimpleImputer()

        X_clean = imputer.fit_transform(X)
        expected = np.array([[1.0, 0.0], [0.0, 3.0]])
        np.testing.assert_array_equal(X_clean, expected)

    def test_custom_fill_value(self):
        """Replace NaN with a custom numeric value."""
        X = np.array([[1.0, np.nan], [np.nan, 3.0]])
        imputer = SimpleImputer(fill_value=-1.0)

        X_clean = imputer.fit_transform(X)
        expected = np.array([[1.0, -1.0], [-1.0, 3.0]])
        np.testing.assert_array_equal(X_clean, expected)

    def test_no_missing_values(self):
        """When there are no NaN, data should be unchanged."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        imputer = SimpleImputer()

        X_clean = imputer.fit_transform(X)
        np.testing.assert_array_equal(X_clean, X)

    def test_all_missing_values(self):
        """All values are NaN, all replaced with fill_value."""
        X = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        imputer = SimpleImputer(fill_value=5.0)
        expected = np.full_like(X, 5.0)

        X_clean = imputer.fit_transform(X)
        np.testing.assert_array_equal(X_clean, expected)

    def test_string_missing_placeholder_default(self):
        """String array with default missing_value=np.nan, it should handle by casting."""

        X_str = np.array([['1.0', 'nan'], ['nan', '3.0']])
        imputer = SimpleImputer(fill_value=0.0)
        X_clean = imputer.fit_transform(X_str)

        expected = np.array([[1.0, 0.0], [0.0, 3.0]])
        np.testing.assert_array_almost_equal(X_clean, expected)

    def test_string_custom_missing_value(self):
        """Replace a custom string placeholder, like 'nan'."""
        X_str = np.array([['1.0', 'nan'], ['nan', '3.0']])
        imputer = SimpleImputer(fill_value='0.0', missing_value='nan')
        X_clean = imputer.fit_transform(X_str)

        expected = np.array([['1.0', '0.0'], ['0.0', '3.0']])
        np.testing.assert_array_equal(X_clean, expected)

    def test_empty_string_missing_value(self):
        """Replace empty strings with a constant."""
        X_str = np.array([['hello', ''], ['', 'world']])
        imputer = SimpleImputer(fill_value='unknown', missing_value='')
        X_clean = imputer.fit_transform(X_str)

        expected = np.array([['hello', 'unknown'], ['unknown', 'world']])
        np.testing.assert_array_equal(X_clean, expected)

    def test_numeric_missing_value_not_nan(self):
        """Replace a specific numeric code like -999 with another value."""
        X = np.array([[1, -999], [-999, 2]], dtype=float)
        imputer = SimpleImputer(fill_value=0, missing_value=-999)
        X_clean = imputer.fit_transform(X)

        expected = np.array([[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(X_clean, expected)

    def test_1d_input(self):
        """Should work with 1D array."""
        X_1d = np.array([1.0, np.nan, 3.0])
        imputer = SimpleImputer(fill_value=0)
        X_clean = imputer.fit_transform(X_1d)

        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(X_clean, expected)

    def test_dtype_preservation_on_numeric(self):
        """Numeric input should retain float dtype after imputation (or be promoted)."""
        X = np.array([[1, np.nan], [np.nan, 2]])
        imputer = SimpleImputer(fill_value=0)

        X_clean = imputer.fit_transform(X)
        self.assertTrue(np.issubdtype(X_clean.dtype, np.floating))

if __name__ == '__main__':
    unittest.main()
