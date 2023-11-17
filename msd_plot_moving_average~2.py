import os
import glob

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
WORKING_DIR = r'C:\Users\pc\Documents\__protein design\SURPASS\Rouse Model\mc002'
OUTPUT_DIR = 'log-log_wrap_moving_average~2'
LOWER_LIMIT = -1000
UPPER_LIMIT = 1000
BOX_SIZE = UPPER_LIMIT - LOWER_LIMIT
WINDOW_SIZE = 5

def moving_average(data, window_size):
    """Compute moving average of given data."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def process_directory(dir):
    """Process a directory and generate the log-log plot."""
    print(f"Processing directory: {dir}")

    # Read cm.dat file in the directory, skipping the first row
    file_path = os.path.join(dir, 'cm.dat')
    data = np.loadtxt(file_path, skiprows=1)

    # Apply periodic boundary conditions
    data = ((data - LOWER_LIMIT) % BOX_SIZE) + LOWER_LIMIT

    # Compute displacement vectors
    displacements = data - data[0, :]

    # Compute squared displacement for each time step
    squared_displacements = np.sum(np.square(displacements), axis=1)

    # Compute MSD for each time step
    msd = squared_displacements

    # Smooth MSD using moving average
    smooth_msd = moving_average(msd, WINDOW_SIZE)

    # Generate plots
    plt.figure()
    total_time_steps = len(smooth_msd) + WINDOW_SIZE - 1
    max_x_range = total_time_steps // 100  # 1/100 of total time steps
    plt.loglog(range(WINDOW_SIZE, len(smooth_msd) + WINDOW_SIZE), smooth_msd, label='Smoothed MSD')
    plt.xlim([WINDOW_SIZE, max_x_range])  # Limit X range
    plt.xlabel('MC Steps (Log Scale)')
    plt.ylabel('Smoothed MSD (Log Scale)')
    plt.title('Mean Square Displacement (Log-Log Plot)')
    plt.legend()
    # Save the plot in OUTPUT_DIR directory
    plt.savefig(os.path.join(OUTPUT_DIR, f'{dir}.png'))
    # Close the current figure to free up memory
    plt.close()

# Main script execution
def main():
    os.chdir(WORKING_DIR)
    # Create 'mean_square_displacement' directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Get all directories with the specified format
    dirs = glob.glob('run*_inner*_outer*_factor*_res*')
    for dir in dirs:
        process_directory(dir)

if __name__ == "__main__":
    main()