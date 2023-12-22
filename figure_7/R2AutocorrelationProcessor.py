import os
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class R2AutocorrelationProcessor:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.r2_values = self.get_r2_values_from_file(dir_path)
        self.autocorrelation = self.get_autocorrelation(self.r2_values)
        self.intersection = self.find_intersection_with_one_over_e(self.autocorrelation)

    def get_r2_values_from_file(self, dir_path):
        file_path = os.path.join(dir_path, 'r2.dat')
        print(f'Now processing file: {file_path}')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")
        r2_values = np.loadtxt(file_path, skiprows=1)
        return r2_values

    def get_autocorrelation(self, r2_array):
        r2_mean = np.mean(r2_array)
        autocorrelation = signal.correlate(r2_array - r2_mean, r2_array - r2_mean, mode='full')
        autocorrelation = autocorrelation[autocorrelation.size // 2:] / autocorrelation[autocorrelation.size // 2]
        return autocorrelation

    def find_intersection_with_one_over_e(self, autocorrelation_array):
        one_over_e = 1 / np.e
        intersections = np.argwhere(np.diff(np.sign(autocorrelation_array - one_over_e))).flatten()
        for idx in intersections:
            if autocorrelation_array[idx] >= one_over_e and autocorrelation_array[idx + 1] <= one_over_e:
                slope = (autocorrelation_array[idx + 1] - autocorrelation_array[idx])
                if slope == 0:
                    return idx
                intersection = idx + (one_over_e - autocorrelation_array[idx]) / slope
                return intersection
        return None

    def plot_autocorrelation(self, plot_file_name):
        plt.figure(figsize=(10, 5))
        plt.plot(self.autocorrelation, label='Autocorrelation')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation of $r^2$ Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_file_name + '.png')
        plt.show()

    def plot_intersection(self, plot_file_name):
        if self.intersection is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(self.intersection, label='Intersection')
            plt.axvline(x=self.intersection, color='r', linestyle='--', label='Intersection (1/e)')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.title('Intersection of Autocorrelation with $1/e$')
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_file_name + '.png')
            plt.show()
        else:
            print("No intersection with 1/e was found.")

    def get_residue_number_from_path(self):
        '''
        Extracts the residue number from the directory path by looking for the
        pattern 'residue' followed by a sequence of digits.
        '''
        match = re.search(r"residue(\d+)", self.dir_path)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("Residue number not found in directory path.")

    def get_autocorrelation_data(self):
        return np.arange(self.autocorrelation.size), self.autocorrelation

    def get_intersection_data(self):
        if self.intersection is not None:
            return self.intersection, 1 / np.e
        else:
            return None, None


if __name__ == '__main__':
    dir_path = r'C:\git\rouse_data\mc001\run000_inner72_outer643_factor67_residue47'
    processor = R2AutocorrelationProcessor(dir_path)

    # To get the residue number:
    residue_number = processor.get_residue_number_from_path()
    print(f"The residue number is: {residue_number}")

    # To get the autocorrelation data:
    x_autocorr, y_autocorr = processor.get_autocorrelation_data()

    # To get the intersection data:
    x_intersection, y_intersection = processor.get_intersection_data()

    # To plot autocorrelation:
    processor.plot_autocorrelation('r2_autocorrelation_plot')

    # To plot intersection:
    processor.plot_intersection('r2_intersection_plot')