import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set global working directory
WORKING_DIR = r'C:\Users\pc\Documents\__protein design\SURPASS\Rouse Model\mc002'
os.chdir(WORKING_DIR)

# Set output directory as a constant
OUTPUT_DIR = 'log-log_wrap'

# Create 'mean_square_displacement' directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Get all directories with the specified format
dirs = glob.glob('run*_inner*_outer*_factor*_res*')

# Define the limits of the simulation box
lower_limit = -1000
upper_limit = 1000
box_size = upper_limit - lower_limit

for dir in dirs:
    # Print the directory being processed
    print(f"Processing directory: {dir}")

    # Read cm.dat file in the directory, skipping the first row
    file_path = os.path.join(dir, 'cm.dat')
    data = np.loadtxt(file_path, skiprows=1)

    # Apply periodic boundary conditions
    data = ((data - lower_limit) % box_size) + lower_limit

    # Compute displacement vectors
    displacements = data - data[0, :]

    # Compute squared displacement for each time step
    squared_displacements = np.sum(np.square(displacements), axis=1)

    # Compute MSD for each time step
    msd = squared_displacements.reshape(-1, 1)

    # Generate plots
    plt.figure()
    plt.loglog(range(1, len(data)+1), msd, label='MSD')  # using loglog plot
    plt.xlabel('MC Steps (Log Scale)')
    plt.ylabel('MSD (Log Scale)')
    plt.title('Mean Square Displacement (Log-Log Plot)')
    plt.legend()

    # Save the plot in OUTPUT_DIR directory
    plt.savefig(f'{OUTPUT_DIR}/{dir}.png')

    # Close the current figure to free up memory
    plt.close()

