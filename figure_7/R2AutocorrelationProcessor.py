import numpy as np
import os
import re

class R2AutocorrelationProcessor:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.r2_values = self.get_r2_values_from_file(dir_path)
        self.auto_correlation = self.calculate_autocorrelation(self.r2_values)
        self.intersection = self.find_intersection_with_one_over_e()
        self.residue_length = self.get_residue_count()

    def get_r2_values_from_file(self, dir_path):
        file_path = os.path.join(dir_path, "r2.dat")
        print(f"Now processing file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")

        with open(file_path, 'r') as file:
            r2_values = np.array([float(line.strip()) for line in file.readlines()[1:]])
        return r2_values

    def calculate_autocorrelation(self, data):
        n = len(data)
        mean = np.mean(data)
        variance = np.var(data, ddof=0)
        autocorr = np.correlate(data - mean, data - mean, mode='full')[n-1:] / (variance * np.arange(n, 0, -1))
        return autocorr

    def find_intersection_with_one_over_e(self):
        one_over_e = 1 / np.e
        for i in range(len(self.auto_correlation) - 1):
            if self.auto_correlation[i] >= one_over_e > self.auto_correlation[i + 1]:
                slope = (self.auto_correlation[i + 1] - self.auto_correlation[i])
                if abs(slope) < 1e-9:  # Prevent division by zero
                    return i
                intersection = i + (one_over_e - self.auto_correlation[i]) / slope
                return intersection
        return None

    def get_residue_count(self):
        match = re.search(r'residue(\d+)', self.dir_path)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("Residue number not found in directory path.")

# Example usage:
# processor = R2AutocorrelationProcessor('/path/to/directory')
# print(processor.r2_values)
# print(processor.auto_correlation)
# print(processor.intersection)
# print(processor.residue_length)