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
        self.test_dir = '../mc001/run000_inner72_outer643_factor67_residue47'

    def tearDown(self):
        pass

    def test_get_r2_values_from_file(self):
        processor = R2AutocorrelationProcessor(self.test_dir)
        r2_values = processor.get_r2_values_from_file(self.test_dir)
        self.assertTrue((r2_values == np.arange(10)**2).all())

    def test_get_autocorrelation(self):
        processor = R2AutocorrelationProcessor(self.test_dir)
        autocorrelation = processor.get_autocorrelation(np.arange(10)**2)
        self.assertIsNotNone(autocorrelation)
        expected_arr = np.array([1.00000, 0.68535, 0.38638, 0.11820, -0.10519, -0.27144, -0.37043, -0.39481, -0.34055, -0.20751])
        # Use numpy's allclose function to compare the arrays within some tolerance.
        self.assertTrue(np.allclose(autocorrelation, expected_arr, atol=1e-5))

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



if __name__ == '__main__':
    unittest.main()