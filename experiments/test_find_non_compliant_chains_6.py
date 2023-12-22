import os
import unittest
import numpy as np
from experiments.find_non_compliant_chains_6 \
    import calculate_autocorrelation, read_r2_values_from_file, find_intersection_with_one_over_e


class TestRouseModelAnalysis(unittest.TestCase):
    # Setup method to prepare any required state, e.g., creating temporary files or directories.
    def setUp(self):
        pass

    # Teardown method to clean up any created resources.
    def tearDown(self):
        pass

    def test_calculate_autocorrelation(self):
        # Test the calculate_autocorrelation function with a known input.
        r_squared_values = [1, 2, 3, 4, 5]
        expected_output = [1, 0.4, 0.04, -0.12, -0.2]  # Example expected output
        autocorrelation = calculate_autocorrelation(r_squared_values)
        # Check if the calculated autocorrelation is close enough to the expected output.
        np.testing.assert_allclose(autocorrelation, expected_output, rtol=1e-5)

    def test_read_r2_values_from_file(self):
        # Test the read_r2_values_from_file function with a temporary file.
        tmp_file_path = 'temp_r2.dat'
        with open(tmp_file_path, 'w') as f:
            f.write("Header line\n")
            f.write("1.0\n2.0\n3.0\n")

        # Read the values using the generator function.
        r2_values = list(read_r2_values_from_file(tmp_file_path))
        expected_values = [1.0, 2.0, 3.0]
        self.assertEqual(r2_values, expected_values)

        # Clean up the temporary file.
        os.remove(tmp_file_path)

    def test_find_intersection_with_one_over_e(self):
        # Test the find_intersection_with_one_over_e function.
        autocorrelation_values = [1.0, 0.7, 0.5, 0.3, 0.1]
        intersection = find_intersection_with_one_over_e(autocorrelation_values)
        expected_intersection = 2.5  # Example expected intersection
        self.assertEqual(intersection, expected_intersection)

    # More tests should be written for other functions like process_directories and compile_results,
    # but they are more complex as they require setting up directory structures and files.
    # For process_directories, one would mock os.listdir and open functions to return
    # predetermined values and check if the final output matches the expected result.

if __name__ == '__main__':
    unittest.main()