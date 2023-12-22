import unittest
import os
import shutil
import tempfile
import os
import re
import math
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch

# Assuming the class R2AutocorrelationProcessor is in a module named processor
import numpy as np

from figure_7.R2AutocorrelationProcessor import R2AutocorrelationProcessor

class TestR2AutocorrelationProcessor(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create a sample r2.dat file
        self.r2_file_path = os.path.join(self.test_dir, 'r2.dat')
        with open(self.r2_file_path, 'w') as f:
            f.write('Time(s) R2\n')  # Header
            for i in range(10):
                f.write(f'{i} {i**2}\n')  # Sample R2 values

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_get_r2_values_from_file(self):
        processor = R2AutocorrelationProcessor(self.test_dir)
        r2_values = processor.get_r2_values_from_file(self.test_dir)
        self.assertTrue((r2_values == np.arange(10)**2).all())

    def test_get_autocorrelation(self):
        processor = R2AutocorrelationProcessor(self.test_dir)
        autocorrelation = processor.get_autocorrelation(np.arange(10)**2)
        self.assertIsNotNone(autocorrelation)
        # More detailed tests should be added here based on the expected autocorrelation values.

    def test_find_intersection_with_one_over_e(self):
        processor = R2AutocorrelationProcessor(self.test_dir)
        # Provide a specific test case if a known autocorrelation array is given
        test_autocorrelation = np.array([1, 0.5, 0.25, 0.125, 0.0625])
        intersection = processor.find_intersection_with_one_over_e(test_autocorrelation)
        self.assertEqual(intersection, 1)  # This is a dummy value for the purpose of this example

    def test_get_residue_number_from_path(self):
        processor = R2AutocorrelationProcessor(self.test_dir)
        # Mock the directory path to simulate the presence of the string 'residue' followed by numbers
        processor.dir_path = 'path_to_residue123'
        residue_number = processor.get_residue_number_from_path()
        self.assertEqual(residue_number, 123)

    def test_plot_autocorrelation(self):
        processor = R2AutocorrelationProcessor(self.test_dir)
        with patch.object(plt, 'show'):
            processor.plot_autocorrelation('test_plot')

    def test_plot_intersection(self):
        processor = R2AutocorrelationProcessor(self.test_dir)
        with patch.object(plt, 'show'):
            processor.plot_intersection('test_plot_intersection')

if __name__ == '__main__':
    unittest.main()