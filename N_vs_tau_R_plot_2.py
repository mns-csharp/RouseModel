# File: N_vs_tau_R_plot_2.py
import glob
import io
import os
import re
from statistics import mean

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('Agg')
import math
from scipy.optimize import curve_fit

SOURCE_PATH = r'C:\git\rouse_data\mc008'
DEST_PATH = r'C:\git\rouse_data\mc008\msd~1'
DAT_FILE = "cm.dat"
TAU_MAX = 100
TEXT_FILE_NAME = "curvature.txt"
HEADERS = 'tau_max, inner, outer, factor, res, curvature\n'
R2_DAT_FILE = "r2.dat"

def load_r2(filename: str) -> np.ndarray:
    """Load R^2 values from a file, skipping the first row (header)."""
    return np.loadtxt(filename, skiprows=1)

def mean_squared_distance(r2: np.ndarray) -> float:
    """Calculate the mean squared distance of an array."""
    return mean(np.square(r2))

def autocorrelation_function(r2: np.ndarray) -> np.ndarray:
    """Compute the autocorrelation function g_R(t) based on the mean-squared end-to-end distance."""
    r0 = np.sqrt(r2[0])  # R(0) is the square root of the first R^2 value
    r_t = np.sqrt(r2)    # R(t) is the square root of each R^2 value
    r_squared_mean = mean_squared_distance(r2)
    g_R = (r0 * r_t) / r_squared_mean
    return g_R

def fit_exponential_decay(g_R: np.ndarray, t: np.ndarray) -> float:
    """Fit g_R(t) to an exponential decay to find the relaxation time tau_R."""
    # Fit g_R(t) = exp(-t / tau_R) to find tau_R
    # Initial guess for tau_R
    tau_R_guess = 1.0
    popt, pcov = curve_fit(lambda t, tau_R: np.exp(-t / tau_R), t, g_R, p0=tau_R_guess)
    tau_R = popt[0]
    return tau_R

def calculate_tau_R(r2: np.ndarray, time_interval: float = 1.0) -> float:
    """Calculate the relaxation time tau_R as the time at which R^2 drops to 1/e of its initial value."""
    r2_initial = r2[0]
    r2_e = r2_initial / math.e
    tau_R_index = next((i for i, r2_value in enumerate(r2) if r2_value < r2_e), None)
    if tau_R_index is not None:
        tau_R = tau_R_index * time_interval
    else:
        tau_R = None  # or any other value you deem appropriate
    return tau_R

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


def plot_log_tauR_vs_log_N(tau_R: np.ndarray, N: np.ndarray) -> io.BytesIO:
    """Plot log(tau_R) against log(N) and return the plot as a BytesIO object."""
    fig, ax = plt.subplots()
    ax.loglog(N, tau_R, marker='o')  # Remove np.log from here
    ax.set_xlabel('N')
    ax.set_ylabel('tau_R')
    ax.set_title('N vs. tau_R in log-log scale)')
    ax.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img

def get_directories(dir_path: str, pattern: str='run*_inner*_outer*_factor*_res*') -> list[str]:
    """This function retrieves all directories in the specified path that match a given pattern."""
    return glob.glob(os.path.join(dir_path, pattern))

def write_image_to_directory(img: io.BytesIO, directory: str, filename: str) -> None:
    """Write image data to a file in the specified directory."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        f.write(img.read())
    #end-of-with
#end-of-function


def process_directories(source_path: str=SOURCE_PATH, dest_path: str=DEST_PATH, dat_file: str=DAT_FILE, r2_dat_file: str=R2_DAT_FILE, time_interval: float = 1.0) -> None:
    """Process all directories in source_path, calculate tau_R for different N, and write results to dest_path."""
    directories = get_directories(source_path)
    tau_R_values = []  # Renamed from log_tau_R
    N_values = []  # Renamed from log_N
    for directory in directories:
        filename = os.path.join(directory, dat_file)
        r2_filename = os.path.join(directory, r2_dat_file)
        if os.path.isfile(filename) and os.path.isfile(r2_filename):
            r2 = load_r2(r2_filename)
            g_R = autocorrelation_function(r2)
            t = np.arange(len(g_R)) * time_interval  # Assuming that the time steps are equally spaced
            tau_R = fit_exponential_decay(g_R, t)
            if tau_R is not None and tau_R > 0.0:
                tau_R_values.append(tau_R)  # Remove math.log from here
                _, _, _, N = extract_values(os.path.basename(directory))
                N_values.append(N ** 2.2)  # Consider the scaling τ_R ≈ N^2.2, remove math.log from here
        else:
            print(f"File {filename} or {r2_filename} does not exist.")
        # end-of-if-else
    # end-of-for-loop
    img = plot_log_tauR_vs_log_N(np.array(tau_R_values), np.array(N_values))  # Use the new name here
    write_image_to_directory(img, dest_path, 'N_vs_tau_R_plot_2.png')


if __name__ == "__main__":
    process_directories()

