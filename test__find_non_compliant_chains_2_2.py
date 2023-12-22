import unittest
from unittest.mock import patch, mock_open
from find_non_compliant_chains_2_2 import DirProcess, R2

# Mock data for testing
mock_directory_list = [
    'run1_inner2_outer3_factor4_residue5',
    'run2_inner3_outer4_factor5_residue6',
    'run3_inner4_outer5_factor6_residue7',
    'invalid_directory1',
    'invalid_directory2'
]

mock_r2_file_content = """# Time R2
0.0 1.0
1.0 2.0
2.0 3.0
"""

class TestDirProcess(unittest.TestCase):
    def test_get_r2_directories(self):
        r2_dirs = DirProcess.get_r2_directories('dummy_path')
        self.assertEqual(len(r2_dirs), 3)  # Should find 3 valid directories
        self.assertIn('run1_inner2_outer3_factor4_residue5', r2_dirs)

class TestR2(unittest.TestCase):
    def test_read_r2_values_from_file(self):
        m = mock_open(read_data=mock_r2_file_content)
        with patch('builtins.open', m):
            r2_values = R2.read_r2_values_from_file('dummy_file_path')
            self.assertEqual(len(r2_values), 3)  # Should read 3 R2 values
            self.assertEqual(r2_values[0], 1.0)  # First value should be 1.0

    def test_calculate_autocorrelation(self):
        r_squared_values = [1, 2, 3, 4, 5]
        autocorrelation = R2.calculate_autocorrelation(r_squared_values)
        self.assertEqual(len(autocorrelation), len(r_squared_values))
        # More detailed tests are needed to verify the correctness of each autocorrelation value

    def test_find_intersection_with_one_over_e(self):
        autocorrelation_values = [1.0, 0.5, 0.2, 0.1]
        intersection = R2.find_intersection_with_one_over_e(autocorrelation_values)
        self.assertIsNotNone(intersection)  # There should be an intersection
        # More detailed tests are needed to verify the correct intersection value

    def test_process_directories(self):
        # This test would be more complex due to its dependence on filesystem and other static methods
        pass

    def test_calculate_standard_deviation(self):
        values = [1, 2, 3, 4, 5]
        stddev = R2.calculate_standard_deviation(values)
        self.assertAlmostEqual(stddev, 1.41421356237, places=5)

if __name__ == '__main__':
    unittest.main()