import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set global working directory
WORKING_DIR = r'C:\Users\pc\Documents\__protein design\SURPASS\Rouse Model\surpass_data_gen'
os.chdir(WORKING_DIR)

# Create 'mean_square_displacement' directory if it doesn't exist
if not os.path.exists('mean_square_displacement'):
    os.makedirs('mean_square_displacement')

# Get all directories with the specified format
dirs = glob.glob('run*_inner*_outer*_factor*_res*')

for dir in dirs:
    # Print the directory being processed
    print(f"Processing directory: {dir}")

    # Read cm.dat file in the directory, skipping the first row
    file_path = os.path.join(dir, 'cm.dat')
    data = np.loadtxt(file_path, skiprows=1)

    # Compute displacement vectors
    displacements = data - data[0, :]

    # Compute squared displacement for each time step
    squared_displacements = np.sum(np.square(displacements), axis=1)

    # Compute MSD for each time step
    msd = np.mean(squared_displacements.reshape(-1, 1), axis=1)

    # Generate plots
    plt.figure()
    plt.plot(range(len(data)), msd, label='MSD')
    plt.xlabel('MC Steps')
    plt.ylabel('MSD')
    plt.title('Mean Square Displacement')
    plt.legend()

    # Save the plot in 'mean_square_displacement' directory
    plt.savefig(f'mean_square_displacement/{dir}.png')

    # Clear the current figure to avoid overlapping of plots in the next iteration
    plt.clf()