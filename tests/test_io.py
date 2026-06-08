"""
test_io.py
Unit tests for io: Load CSV
"""

import sys
import unittest
import numpy as np
import tempfile
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from numcompute import io as io_module


class TestLoadCSV(unittest.TestCase):
    """Tests for io.load_csv."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test.csv")

    def tearDown(self):
        # Clean up temporary files
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def _write_csv(self, content):
        with open(self.csv_path, 'w') as f:
            f.write(content)

    def test_basic_numeric(self):
        self._write_csv("a,b,c\n1,2,3\n4,5,6")
        header, data = io_module.load_csv(self.csv_path)
        self.assertTrue(np.array_equal(header, np.array(['a', 'b', 'c'])))
        self.assertTrue(np.array_equal(data, np.array([[1., 2., 3.], [4., 5., 6.]])))
        self.assertEqual(data.dtype, float)

    def test_with_header_only(self):
        self._write_csv("col1,col2")
        header, data = io_module.load_csv(self.csv_path)
        self.assertTrue(np.array_equal(header, np.array(['col1', 'col2'])))
        self.assertEqual(data.size, 0)
        self.assertEqual(data.shape, (0, 2))

    def test_empty_file(self):
        self._write_csv("")
        header, data = io_module.load_csv(self.csv_path)
        self.assertEqual(header.size, 0)
        self.assertEqual(data.size, 0)

    def test_missing_values(self):
        self._write_csv("x,y\n1,\n,2")
        header, data = io_module.load_csv(self.csv_path, fill_missing_value='nan')
        self.assertTrue(np.isnan(data[0, 1]))
        self.assertTrue(np.isnan(data[1, 0]))
        self.assertEqual(data[0, 0], 1.0)
        self.assertEqual(data[1, 1], 2.0)

    def test_custom_delimiter(self):
        self._write_csv("a|b|c\n1|2|3")
        header, data = io_module.load_csv(self.csv_path, delimiter='|')
        self.assertTrue(np.array_equal(header, np.array(['a', 'b', 'c'])))
        self.assertTrue(np.array_equal(data, np.array([[1., 2., 3.]])))

    def test_string_data(self):
        self._write_csv("name,city\nAlice,NYC\nBob,LA")
        header, data = io_module.load_csv(self.csv_path)
        self.assertEqual(data.dtype.kind, 'U')  # string type
        self.assertTrue(np.array_equal(header, np.array(['name', 'city'])))
        self.assertEqual(data[1, 1], 'LA')

    def test_single_row_data(self):
        self._write_csv("f1,f2\n10,20")
        header, data = io_module.load_csv(self.csv_path)
        self.assertEqual(data.shape, (1, 2))
        self.assertEqual(data[0, 0], 10.0)


class TestGetColumn(unittest.TestCase):
    """Tests for io.get_column."""

    def test_existing_column_numeric(self):
        header = np.array(['a', 'b', 'c'])
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        col = io_module.get_column('b', data, header)
        np.testing.assert_array_equal(col, np.array([2., 5.]))

    def test_existing_column_string(self):
        header = np.array(['name', 'age'])
        data = np.array([['Alice', '30'], ['Bob', '25']])
        col = io_module.get_column('name', data, header)
        np.testing.assert_array_equal(col, np.array(['Alice', 'Bob']))

    def test_column_not_found(self):
        header = np.array(['x', 'y'])
        data = np.array([[1, 2]])
        with self.assertRaises(IndexError):
            io_module.get_column('z', data, header)


if __name__ == '__main__':
    unittest.main()