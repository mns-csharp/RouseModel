import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List

# Model a 3D point
class Point3d:
    def __init__(self, xx: float, yy: float, zz: float):
        self.x = xx
        self.y = yy
        self.z = zz

# Model the bounds of a box
class BoxBounds:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

# Wrap a 3D point within box bounds if it exceeds them
def wrap_position(pos: Point3d, bounds: BoxBounds) -> Point3d:
    # For each dimension check if point exceeds the bounds and wrap it if necessary
    for dim in ['x', 'y', 'z']:
        if getattr(pos, dim) < bounds.lower:
            setattr(pos, dim, bounds.upper - (bounds.lower - getattr(pos, dim)))
        if getattr(pos, dim) > bounds.upper:
            setattr(pos, dim, bounds.lower + (getattr(pos, dim) - bounds.upper))
    return pos

# Parse a file and return a list of 3D points
def parse_cm_file(file_path: str) -> List[Point3d]:
    atoms = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                x, y, z = map(float, line.split())
                atoms.append(Point3d(x, y, z))
    return atoms

# Calculate the mean square displacement for each 3D point from the first one
def calculate_msd(atoms: List[Point3d]) -> List[float]:
    msd = [0]
    for i in range(1, len(atoms)):
        dx, dy, dz = (getattr(atoms[i], dim) - getattr(atoms[0], dim) for dim in ['x', 'y', 'z'])
        msd.append(dx**2 + dy**2 + dz**2)
    return msd

# Calculate the diffusion coefficient from the mean square displacement
def calculate_diffusion_coefficient(msd: List[float], time_step: float) -> float:
    return msd[-1] / (6 * len(msd) * time_step)

# Plot a list of values with labels and title, save it to a file
def plot_data(steps: List[int], values: List[float], x_label: str, y_label: str, title: str, output_dir: str, filename: str) -> None:
    plt.figure()
    plt.plot(steps, values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Process a directory: parse a file, calculate MSD and diffusion, save plots
def process_directory(directory: str, cm_file_name: str, time_step: float) -> float:
    cm_file = os.path.join(directory, cm_file_name)
    bounds = BoxBounds(-1000, 1000)  # Update bounds
    atoms = [wrap_position(atom, bounds) for atom in parse_cm_file(cm_file)]
    msd = calculate_msd(atoms)
    output_dir = os.path.join(directory, 'mean_square_displacement')
    os.makedirs(output_dir, exist_ok=True)
    # Plot MSD
    steps = list(range(len(atoms)))
    plot_data(steps, msd, 'Monte Carlo Steps', 'Mean Square Displacement', 'Mean Square Displacement', output_dir, 'msd.png')
    return calculate_diffusion_coefficient(msd, time_step)

# Process multiple directories
def process_directories(root_directory: str, cm_file_name: str, time_step: float) -> None:
    directories = [name for name in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, name)) and name.startswith('run')]
    chain_lengths = []
    diffusion_coefficients = []
    for directory in directories:
        chain_length = int(directory[3:])  # Assuming the directory name is "run" followed by the chain length
        diffusion_coefficient = process_directory(os.path.join(root_directory, directory), cm_file_name, time_step)
        chain_lengths.append(chain_length)
        diffusion_coefficients.append(diffusion_coefficient)
    output_dir = os.path.join(root_directory, 'mean_square_displacement')
    plot_data(chain_lengths, diffusion_coefficients, 'Chain Length', 'Diffusion Coefficient', 'Diffusion Coefficient vs Chain Length', output_dir, 'diffusion.png')

def main():
    root_directory = '.'  # Path to the parent directory of the "run" directories
    cm_file_name = 'cm.dat'  # Define the file name
    time_step = 1  # Define the time step
    process_directories(root_directory, cm_file_name, time_step)

if __name__ == "__main__":
    main()
    
