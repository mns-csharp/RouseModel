# File: N_vs_tau_R_plot.py
import glob
import io
import os
import re

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import math


SOURCE_PATH = r'C:\git\rouse_data\mc010'
DEST_PATH = r'C:\git\rouse_data\mc010'
DAT_FILE = "cm.dat"
TAU_MAX = 100.0
TEXT_FILE_NAME = "curvature.txt"
HEADERS = 'tauMax, inner, outer, factor, res, curvature\n'
R2_DAT_FILE = "r2.dat"
SOME_THRESHOLD = 10^3


def load_r2(filename: str) -> np.ndarray:
    """Load R^2 values from a file, skipping the first row (header)."""
    return np.loadtxt(filename, skiprows=1)

'''
def calculateTauR(r2: np.ndarray, time_interval: float = 1.0) -> float:
    """Calculate the relaxation time tau_R as the time at which R^2 drops to 1/e of its initial value."""
    r2_initial = r2[0]
    r2_e = r2_initial / math.e
    tau_R_index = next((i for i, r2_value in enumerate(r2) if r2_value < r2_e), None)
    if tau_R_index is not None:
        tau_R = tau_R_index * time_interval
    else:
        tau_R = None  # or any other value you deem appropriate
    return tau_R
'''

def calculate_tau_R(r2: np.ndarray, time_interval: float = 1.0) -> float:
    """Calculate the relaxation time tau_R as the time at which R^2 drops to 1/e of its initial value."""
    r2_initial = r2[0]
    r2_e = r2_initial / math.e
    r2_below_e_count = 0
    tau_R_index = None
    for i, r2_value in enumerate(r2):
        if r2_value < r2_e:
            r2_below_e_count += 1
            if r2_below_e_count >= SOME_THRESHOLD:  # some threshold of consecutive measurements
                tau_R_index = i
                break
        else:
            r2_below_e_count = 0  # reset count if r2_value rises above r2_e
    #end-of-for
    tau_R = tau_R_index * time_interval if tau_R_index is not None else None
    return tau_R

def plot_N_vs_tau_R(N: np.ndarray, tau_R: np.ndarray) -> io.BytesIO:
    """Plot log(tau_R) against log(N) and return the plot as a BytesIO object."""
    fig, ax = plt.subplots()
    ax.loglog(N, tau_R, marker='o')
    ax.set_xlabel('N')
    ax.set_ylabel('tau_R')
    ax.set_title('N vs. tau_R in Logarithmic Scale')
    ax.grid(True)
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
        print(f"Error with directory_path {input_string}: {str(e)}")
    #end-of-try-except
#end-of-function

def write_image_to_directory(img: io.BytesIO, directory: str, filename: str) -> None:
    """Write image data to a file in the specified directory_path."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        f.write(img.read())
    #end-of-with
#end-of-function

def process_directory(directory: str, dat_file: str, r2_dat_file: str, time_interval: float) -> list:
    """Process a single directory_path and return a list of [N, tau_R] or None if files do not exist."""
    filename = os.path.join(directory, dat_file)
    r2_filename = os.path.join(directory, r2_dat_file)
    if os.path.isfile(filename) and os.path.isfile(r2_filename):
        r2 = load_r2(r2_filename)
        tau_R = calculate_tau_R(r2, time_interval)
        if tau_R is not None and tau_R > 0.0:
            _, _, _, N = extract_values(os.path.basename(directory))
            return [N, tau_R]
        else:
            return None
    else:
        print(f"File {filename} or {r2_filename} does not exist.")
        return None

def process_directories(source_path: str=SOURCE_PATH, dest_path: str=DEST_PATH, dat_file: str=DAT_FILE, r2_dat_file: str=R2_DAT_FILE, time_interval: float = 1.0) -> None:
    """Process all directories in sourcePath, calculate tau_R for different N, and write results to dest_path."""
    directories = get_directories(source_path)
    n_vs_tau_list = []
    for directory in directories:
        result = process_directory(directory, dat_file, r2_dat_file, time_interval)
        if result is not None:
            n_vs_tau_list.append(result)
    n_vs_tau_list.sort(key=lambda x: x[0])
    first_column, second_column = zip(*n_vs_tau_list)
    img = plot_N_vs_tau_R(N=np.array(first_column), tau_R=np.array(second_column))
    write_image_to_directory(img, dest_path, 'N_vs_tau_R_plot.png')

if __name__ == "__main__":
    process_directories()

