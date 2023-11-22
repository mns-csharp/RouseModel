# File: d_vs_n_plot.py
import glob
import io
import os
import re

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SOURCE_PATH = r'C:\git\rouse_data\mc008'
DEST_PATH = r'C:\git\rouse_data\mc008\msd~1'
DAT_FILE = "cm.dat"
TAU_MAX = 100
TEXT_FILE_NAME = "curvature.txt"
HEADERS = 'tau_max, inner, outer, factor, res, curvature\n'


def load_CM_positions(filename: str) -> np.ndarray:
    """Load positions from a file, skipping the first row (header)."""
    return np.loadtxt(filename, skiprows=1)


# https://www.scm.com/doc/Tutorials/MolecularDynamicsAndMonteCarlo/BatteriesDiffusionCoefficients.html
def clear_text_file(filepath: str) -> None:
    """Clear the contents of a text file."""
    if os.path.exists(filepath):
        with open(filepath, 'w') as file:
            file.write('')
        #end-of-with
    #end-of-if
#end-of-function

from scipy.stats import linregress
def calculate_slope(tau: np.ndarray, MSD: np.ndarray) -> float:
    """Calculate slope of the MSD-vs-tau curve using linear regression."""
    slope, _, _, _, _ = linregress(x=MSD, y=tau)
    return slope


def get_lag_times(tau_max: int) -> np.ndarray:
    """Generate lag times based on tau_max."""
    return np.arange(1, tau_max + 1)


def calculate_MSD(positions: np.ndarray, tau_max: int=0) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Mean Square Displacement (MSD) for each lag time."""
    if tau_max != 0:
        tau_max = int(len(positions) / tau_max)
    else:
        tau_max = len(positions) - 1
    tau = get_lag_times(tau_max)
    MSD = np.empty_like(tau, dtype=float)
    for i in range(1, tau_max + 1):
        displacements = positions[i:] - positions[:-i]
        squared_displacements = np.sum(displacements**2, axis=1)
        MSD[i - 1] = np.mean(squared_displacements) if squared_displacements.size > 0 else np.nan
    return tau, MSD


def calculate_D(tau: np.ndarray, MSD: np.ndarray, d: int) -> float:
    """Calculate the diffusion coefficient from the MSD and delay time."""
    slope: float = calculate_slope(tau, MSD)
    D = slope / (2 * d)
    return D


def write_image_to_directory(img: io.BytesIO, directory: str, filename: str) -> None:
    """Write image data to a file in the specified directory."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        f.write(img.read())
    #end-of-with
#end-of-function


def plot_diffusion_vs_length(diffusion_coefficients: np.ndarray, lengths: np.ndarray) -> io.BytesIO:
    """Plot the diffusion coefficients against the polymer lengths on a log-log scale and return the plot as a BytesIO object."""
    plt.figure()
    plt.loglog(lengths, diffusion_coefficients, 'o')
    plt.xlabel('Polymer length N')
    plt.ylabel('Diffusion coefficient D')
    plt.title('Diffusion coefficient vs. Polymer length (log-log scale)')
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img


def get_directories(dir_path: str, pattern: str='run*_inner*_outer*_factor*_res*') -> list[str]:
    """This function retrieves all directories in the specified path that match a given pattern."""
    return glob.glob(os.path.join(dir_path, pattern))


def extract_values(input_string: str) -> tuple[int, int, int, int]:
    """Extract values from a formatted string."""
    try:
        match = re.search(r'run\d+_inner(\d+)_outer(\d+)_factor(\d+)_residue(\d+)', input_string)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        else:
            raise ValueError("Input string does not match required format.")
        #end-of-if-else
    except ValueError as e:
        print(f"Error with directory {input_string}: {str(e)}")
    #end-of-try-except
#end-of-function


def process_directories(source_path: str=SOURCE_PATH, dest_path: str=DEST_PATH, dat_file: str=DAT_FILE) -> None:
    """Process all directories in source_path, calculate MSD and curvature, and write results to dest_path."""
    print("START")
    clear_text_file(os.path.join(dest_path, TEXT_FILE_NAME))
    directories = get_directories(source_path)
    lengths = []
    D = []
    for directory in directories:
        filename = os.path.join(directory, dat_file)
        if os.path.isfile(filename):
            positions = load_CM_positions(filename)
            tau, MSD = calculate_MSD(positions, TAU_MAX)
            # compute diffusion coefficient and length
            D.append(calculate_D(MSD, tau, d=3))
            _, _, _, length = extract_values(os.path.basename(directory))
            lengths.append(length)
        else:
            print(f"File {filename} does not exist.")
        # end-of-if-else
    #end-of-for-loop
    img = plot_diffusion_vs_length(np.array(D), np.array(lengths))
    write_image_to_directory(img, dest_path, 'D_vs_N__plot.png')
    print("END")
#end-of-function


if __name__ == "__main__":
    process_directories()
