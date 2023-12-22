import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Pre-compile the regular expression pattern.
DIR_PATTERN = re.compile(r'run\d+_inner\d+_outer\d+_factor\d+_residue(?P<residue>\d+)', re.IGNORECASE)

# Use NumPy functions for autocorrelation to optimize calculation.
def calculate_autocorrelation(r_squared_values):
    r_squared_values = np.array(r_squared_values)
    N = r_squared_values.size
    mean = np.mean(r_squared_values)
    autocorrelation = np.correlate(r_squared_values - mean, r_squared_values - mean, mode='full')[N-1:] / (N * np.var(r_squared_values))
    return autocorrelation / autocorrelation[0]

# Use a generator to read large files efficiently.
def read_r2_values_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line
        for line in file:
            yield float(line.strip())

# Cache the one_over_e value outside the loop.
def find_intersection_with_one_over_e(autocorrelation_values):
    one_over_e = 1 / math.e
    previous_value = autocorrelation_values[0]
    for i, value in enumerate(autocorrelation_values[1:], 1):
        if previous_value >= one_over_e >= value:
            slope = (value - previous_value)
            if slope == 0:
                return i - 1
            intersection = (i - 1) + (one_over_e - previous_value) / slope
            return intersection
        previous_value = value
    return None

# Optimize the process_directories function to reduce memory usage and avoid redundant file access.
def process_directories(directory_path, iterations=5):
    y_points_all_iterations = {}
    
    for subdirectory in filter(DIR_PATTERN.match, os.listdir(directory_path)):
        residue_value = int(DIR_PATTERN.match(subdirectory).group('residue'))
        y_points_all_iterations[residue_value] = []
    
    for iteration in range(iterations):
        for subdirectory, intersections in y_points_all_iterations.items():
            file_path = os.path.join(directory_path, str(subdirectory), 'r2.dat')
            try:
                r2_values = list(read_r2_values_from_file(file_path))  # Only convert to a list here
                autocorrelation_values = calculate_autocorrelation(r2_values)
                intersection = find_intersection_with_one_over_e(autocorrelation_values)
                if intersection is not None:
                    intersections.append(intersection)
            except Exception as ex:
                print(f"An exception occurred during iteration {iteration} for {subdirectory}: {ex}")
    
    return compile_results(y_points_all_iterations)

# Vectorize the results compilation process.
def compile_results(y_points_all_iterations):
	x_points = np.fromiter(y_points_all_iterations.keys(), dtype=int)
	y_points_mean = np.array([np.mean(vals) if vals else np.nan for vals in y_points_all_iterations.values()])
	y_points_stddev = np.array([np.std(vals, ddof=1) if len(vals) > 1 else np.nan for vals in y_points_all_iterations.values()])
	
	# Only consider non-NaN values for regression
	valid_indices = ~np.isnan(y_points_mean)
	x_points_sorted = x_points[valid_indices]
	y_points_mean_sorted = y_points_mean[valid_indices]
	y_points_stddev_sorted = y_points_stddev[valid_indices]
	
	# Perform linear regression using scipy's linregress function
	slope, intercept, r_value, p_value, std_err = linregress(x_points_sorted, y_points_mean_sorted)
	
	return x_points_sorted, y_points_mean_sorted, y_points_stddev_sorted, slope, intercept, r_value, p_value, std_err


# Plot the results with error bars
def plot_results(x_points, y_points_mean, y_points_stddev, slope, intercept):
	plt.errorbar(x_points, y_points_mean, yerr=y_points_stddev, fmt='o', label='Data points')
	plt.plot(x_points, intercept + slope * x_points, 'r', label=f'Fit: slope={slope:.2f}, intercept={intercept:.2f}')
	plt.xlabel('Residue number')
	plt.ylabel('Intersection with 1/e')
	plt.legend()
	plt.show()


if __name__ == "__main__":
    directory_paths = [  # Corrected variable name to match the usage below
        "/home/mohammad/rouse_data/mc006",
        "/home/mohammad/rouse_data/mc007",
        "/home/mohammad/rouse_data/mc008",
        "/home/mohammad/rouse_data/mc009",
        "/home/mohammad/rouse_data/mc010"
    ]
    iterations = 5  # Specify the number of iterations if needed

    # Assuming we want to compile results from all the directories into a single plot
    all_x_points, all_y_points_mean, all_y_points_stddev = [], [], []
    for directory_path in directory_paths:
        # Process each directory and get the compiled results.
        x_points, y_points_mean, y_points_stddev, slope, intercept, r_value, p_value, std_err = process_directories(directory_path, iterations)
        all_x_points.extend(x_points)
        all_y_points_mean.extend(y_points_mean)
        all_y_points_stddev.extend(y_points_stddev)
    
    # Assuming that you want to combine all data points and then plot
    all_x_points = np.array(all_x_points)
    all_y_points_mean = np.array(all_y_points_mean)
    all_y_points_stddev = np.array(all_y_points_stddev)
    
    # Perform linear regression on the combined data
    valid_indices = ~np.isnan(all_y_points_mean)
    combined_x_points_sorted = all_x_points[valid_indices]
    combined_y_points_mean_sorted = all_y_points_mean[valid_indices]
    combined_y_points_stddev_sorted = all_y_points_stddev[valid_indices]
    combined_slope, combined_intercept, combined_r_value, combined_p_value, combined_std_err = linregress(combined_x_points_sorted, combined_y_points_mean_sorted)
    
    # Plot the combined results.
    plot_results(combined_x_points_sorted, combined_y_points_mean_sorted, combined_y_points_stddev_sorted, combined_slope, combined_intercept)
