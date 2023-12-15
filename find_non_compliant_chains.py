import os
import re
import math
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt


class R2:
    @staticmethod
    def calculate_autocorrelation(r_squared_values):
        N = len(r_squared_values)
        mean = sum(r_squared_values) / N

        autocorrelation = []

        denominator = sum((v - mean) ** 2 for v in r_squared_values)

        for lag in range(N):
            numerator = sum((r_squared_values[i] - mean) * (r_squared_values[i + lag] - mean) for i in range(N - lag))
            autocorrelation.append(numerator / denominator)

        return autocorrelation

    @staticmethod
    def read_r2_values_from_file(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")

        with open(file_path, 'r') as file:
            lines = file.readlines()

        r2_values = []
        for line in lines[1:]:
            try:
                value = float(line.strip())
                r2_values.append(value)
            except ValueError:
                raise ValueError("One or more lines could not be parsed into floats.")

        return r2_values

    @staticmethod
    def find_intersection_with_one_over_e(autocorrelation_values):
        one_over_e = 1 / math.e
        for i in range(len(autocorrelation_values) - 1):
            if autocorrelation_values[i] >= one_over_e and autocorrelation_values[i + 1] <= one_over_e:
                slope = (autocorrelation_values[i + 1] - autocorrelation_values[i])
                if slope == 0:
                    return i
                intersection = i + (one_over_e - autocorrelation_values[i]) / slope
                return intersection
        return None

    @staticmethod
    def process_directories(directory_path):
        x_points = []
        y_points = []

        dir_pattern = re.compile(r'run\d+_inner\d+_outer\d+_factor\d+_residue(?P<residue>\d+)', re.IGNORECASE)

        for subdirectory in os.listdir(directory_path):
            match = dir_pattern.match(subdirectory)
            if match:
                residue_value = float(match.group('residue'))
                file_path = os.path.join(directory_path, subdirectory, 'r2.dat')
                try:
                    r2_values = R2.read_r2_values_from_file(file_path)
                    autocorrelation_values = R2.calculate_autocorrelation(r2_values)
                    intersection = R2.find_intersection_with_one_over_e(autocorrelation_values)
                    if intersection is not None:
                        x_points.append(residue_value)
                        y_points.append(intersection)
                except Exception as ex:
                    print(f"An exception occurred: {ex}")

        combined_list = sorted(zip(x_points, y_points), key=lambda item: item[0])

        x_points, y_points = zip(*combined_list)

        return x_points, y_points

import math

def linear_fit(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x_squared = sum(xi ** 2 for xi in x)

    # Calculate the slope (b) and intercept (a) for y = a + bx
    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    a = (sum_y - b * sum_x) / n

    return a, b

def calculate_deviation(y, y_fit):
    deviations = [abs((y_i - y_fit_i) / y_fit_i) for y_i, y_fit_i in zip(y, y_fit)]
    return deviations

def button4_click():
    try:
        directory_path = "C:/git/rouse_data/mc004"

        x, y = R2.process_directories(directory_path)
        log_x = [math.log(xi) for xi in x]
        log_y = [math.log(yi) for yi in y]

        # Perform a linear fit on the log-log transformed data
        a, b = linear_fit(log_x, log_y)

        # Convert the fitted line coefficients back to the original scale
        a_exp = math.exp(a)
        fitted_line = [a_exp * xi**b for xi in x]

        plt.figure()
        plt.scatter(x, y, label='Data')
        plt.plot(x, fitted_line, color='red', label=f'Fitted line (slope={b:.2f})')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("N Values")
        plt.ylabel("tau_R Values")
        plt.title("tau_R vs N plot")
        plt.legend()
        plt.grid(True)
        # plt.savefig('N_vs_tau_R_plot_4.png')
        # plt.show()

        # Checking the compliance with the Rouse model
        expected_slope = 2.0
        if abs(b - expected_slope) > 0.1:  # Allowing some tolerance
            print(f"The chain lengths do not comply with the Rouse theory (slope={b:.2f}).")
        else:
            print(f"The chain lengths comply with the Rouse theory (slope={b:.2f}).")

        # Identifying significant deviations
        deviations = calculate_deviation(y, fitted_line)
        threshold = 0.1  # Set a threshold for significant deviation
        significant_deviations = [i for i, dev in enumerate(deviations) if dev > threshold]
        if significant_deviations:
            print("Significant deviations found at chain lengths:")
            for index in significant_deviations:
                print(f"N = {x[index]}, tau_R = {y[index]}, Fitted tau_R = {fitted_line[index]}")

    except Exception as ex:
        print(f"An exception occurred: {ex}")

# Trigger the plot function
button4_click()