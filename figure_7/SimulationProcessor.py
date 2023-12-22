import os
import re
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

import numpy as np

from R2AutocorrelationProcessor import R2AutocorrelationProcessor


class SimulationProcessor:
    DIR_PATTERN = r'run\d+_inner\d+_outer\d+_factor\d+_residue(?P<residue>\d+)'

    def __init__(self, root_directory):
        self.root_directory = root_directory
        self.intersection_data = []

    def get_r2_directories(self):
        dir_pattern = re.compile(self.DIR_PATTERN, re.IGNORECASE)
        r2_directories = []
        for subdirectory in os.listdir(self.root_directory):
            match = dir_pattern.match(subdirectory)
            if match:
                r2_directories.append(subdirectory)
        return r2_directories

    def process_simulations(self):
        # Get a list of r2 directories
        r2_directories = self.get_r2_directories()

        # Process each directory
        for dir_name in r2_directories:
            dir_path = os.path.join(self.root_directory, dir_name)
            try:
                # Initialize the R2AutocorrelationProcessor for the current directory
                print(r'Now processing residue: {dir_path}')
                r2_processor = R2AutocorrelationProcessor(dir_path)
                # Get the intersection data from the current processor
                x_intersection, y_intersection = r2_processor.get_intersection_data()
                if x_intersection is not None and y_intersection is not None:
                    self.intersection_data.append((x_intersection, y_intersection))
            except FileNotFoundError:
                print(f"r2.dat file not found in directory: {dir_path}")
            except ValueError as ve:
                print(ve)

    def get_intersection_data(self):
        # Return the collected intersection data
        return self.intersection_data

    def plot_intersection_data_linear(self):
        if not self.intersection_data:
            print("No intersection data to plot.")
            return

        plt.figure(figsize=(10, 5))
        for idx, (x_inter, y_inter) in enumerate(self.intersection_data):
            plt.plot(x_inter, y_inter, 'o', label=f'Intersection {idx+1}')

        plt.axhline(y=1/np.e, color='r', linestyle='--', label='1/e Line')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Intersection of Autocorrelation with $1/e$')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_intersection_data_loglog(self):
        if not self.intersection_data:
            print("No intersection data to plot.")
            return
        plt.figure(figsize=(10, 5))
        for idx, (x_inter, y_inter) in enumerate(self.intersection_data):
            plt.loglog(x_inter, y_inter, 'o', label=f'Intersection {idx+1}')  # Log-log plot
        plt.axhline(y=1/np.e, color='r', linestyle='--', label='1/e Line')
        plt.xlabel('Log(Lag)')
        plt.ylabel('Log(Autocorrelation)')
        plt.title('Log-Log Plot of Intersection of Autocorrelation with $1/e$')
        plt.legend()
        plt.grid(True, which="both", ls="--")  # Grid enabled for both major and minor ticks
        plt.show()

# Example of using the SimulationProcessor class
if __name__ == '__main__':
    root_dir = r'C:\git\rouse_data\mc001'
    sim_processor = SimulationProcessor(root_dir)
    sim_processor.process_simulations()
    sim_processor.plot_intersection_data_linear()
    sim_processor.plot_intersection_data_loglog()