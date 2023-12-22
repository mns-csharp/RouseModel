import os
import re
import numpy as np
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import sem

class R2:
    @staticmethod
    def calculate_autocorrelation(r_squared_values):
        N = len(r_squared_values)
        mean = np.mean(r_squared_values)
        autocorrelation = np.correlate(r_squared_values - mean, r_squared_values - mean, mode='full') / np.sum((r_squared_values - mean) ** 2)
        return autocorrelation[N-1:]

    @staticmethod
    def read_r2_values_from_file(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")
        r2_values = np.loadtxt(file_path, skiprows=1)
        return r2_values

    @staticmethod
    def find_intersection_with_one_over_e(autocorrelation_values):
        one_over_e = 1 / np.e
        intersections = np.argwhere(np.diff(np.sign(autocorrelation_values - one_over_e))).flatten()
        if intersections.size == 0:
            return None
        return intersections[0]

    @staticmethod
    def process_directories(directory_path):
        dir_pattern = re.compile(r'run\d+_inner\d+_outer\d+_factor\d+_residue(?P<residue>\d+)', re.IGNORECASE)
        residues = {}
        for subdirectory in os.listdir(directory_path):
            match = dir_pattern.match(subdirectory)
            if match:
                residue_value = int(match.group('residue'))
                file_path = os.path.join(directory_path, subdirectory, 'r2.dat')
                try:
                    r2_values = R2.read_r2_values_from_file(file_path)
                    autocorrelation_values = R2.calculate_autocorrelation(r2_values)
                    intersection = R2.find_intersection_with_one_over_e(autocorrelation_values)
                    if intersection is not None:
                        if residue_value not in residues:
                            residues[residue_value] = []
                        residues[residue_value].append(intersection)
                except Exception as ex:
                    print(f"An exception occurred for {subdirectory}: {ex}")

        # Convert dictionary to sorted lists
        sorted_residue_keys = sorted(residues.keys())
        mean_intersections = [np.mean(residues[res]) for res in sorted_residue_keys]
        stddev_intersections = [sem(residues[res]) if len(residues[res]) > 1 else 0 for res in sorted_residue_keys]

        return sorted_residue_keys, mean_intersections, stddev_intersections

def main():
    directory_path = "C:/git/rouse_data/mc010"
    # Get x, y, and y_err values from the process_directories function
    x, y, y_err = R2.process_directories(directory_path)

    # Perform log transformation on x and y
    x = np.array(x)
    y = np.array(y)
    y_err = np.array(y_err)
    log_x = np.log(x)
    log_y = np.log(y)
    valid = ~np.isnan(log_y)

    # Perform a linear fit on the log-log transformed data
    model = LinearRegression()
    model.fit(log_x[valid].reshape(-1, 1), log_y[valid])
    a, b = model.intercept_, model.coef_[0]

    # Plotting
    plt.errorbar(x, y, yerr=y_err, fmt='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Residue')
    plt.ylabel('Intersection with 1/e')
    plt.title('Log-Log Plot of Residue vs. Intersection with 1/e')
    plt.plot(x, np.exp(a) * x**b, label=f'Fit: y = exp({a:.2f}) * x^{b:.2f}')

    # Additional code for the reference line with slope 1.0
    # We'll use the same x values for our line
    ref_y = x  # Since on a log-log scale, a slope of 1 implies y = x
    plt.plot(x, ref_y, 'b--', label='Reference line (slope=1)')  # Dotted blue line

    plt.legend()
    plt.savefig('find_non_comliant_chains_3.png')
    plt.show()

if __name__ == '__main__':
    main()