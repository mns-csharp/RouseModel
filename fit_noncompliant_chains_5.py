import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
        r2_values = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the first line (header)
            for line in file:
                value = float(line.strip())
                r2_values.append(value)
        return np.array(r2_values)

    @staticmethod
    def find_intersection_with_one_over_e(autocorrelation_values):
        one_over_e = 1 / np.e
        intersections = np.argwhere(np.diff(np.sign(autocorrelation_values - one_over_e))).flatten()
        if intersections.size == 0:
            return None
        return intersections[0]

    @staticmethod
    def process_directory(directory_path):
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
        return residues

def main():
    directories = [
        "C:/git/rouse_data/mc006",
        "C:/git/rouse_data/mc007",
        "C:/git/rouse_data/mc008",
        "C:/git/rouse_data/mc009",
        "C:/git/rouse_data/mc010"
    ]

    combined_residues = {}

    for directory_path in directories:
        residues = R2.process_directory(directory_path)
        for res, values in residues.items():
            if res not in combined_residues:
                combined_residues[res] = []
            combined_residues[res].extend(values)

    sorted_residue_keys = sorted(combined_residues.keys())
    mean_intersections = [np.mean(combined_residues[res]) for res in sorted_residue_keys]
    stddev_intersections = [np.std(combined_residues[res], ddof=1) if len(combined_residues[res]) > 1 else 0 for res in sorted_residue_keys]

    x = np.array(sorted_residue_keys)
    y = np.array(mean_intersections)
    y_err = np.array(stddev_intersections)
    log_x = np.log(x)
    log_y = np.log(y)
    valid = ~np.isnan(log_y)
    model = LinearRegression()
    model.fit(log_x[valid].reshape(-1, 1), log_y[valid])
    a, b = model.intercept_, model.coef_[0]

    plt.errorbar(x, y, yerr=y_err, fmt='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Residue Length (N)')
    plt.ylabel('tau_R')
    plt.title('Log-Log Plot of Residue vs. Intersection with 1/e')

    fit_y = np.exp(a) * x**b
    plt.plot(x, fit_y, label=f'Fit: y = exp({a:.2f}) * x^{b:.2f}')

    ref_y = x
    plt.plot(x, ref_y, 'b--', label='Reference line (slope=1)')

    plt.legend()
    plt.savefig('find_non_comliant_chains_5.png')
    plt.show()

if __name__ == '__main__':
    main()