# File: relaxation_time_vs_N.py
import glob
import io
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


SOURCE_PATH = r'C:\git\rouse_data\mc010'
DEST_PATH = r'C:\git\rouse_data\mc010'
DAT_FILE = "cm.dat"
TAU_MAX = 100
TEXT_FILE_NAME = "curvature.txt"
HEADERS = 'tauMax, inner, outer, factor, res, curvature\n'
PATTERN = r'run\d+_inner(\d+)_outer(\d+)_factor(\d+)_residue(\d+)'
PATTERN2 ='run*_inner*_outer*_factor*_res*'
R2_FILE = "r2.dat"

def get_directories(dir_path: str, pattern: str=PATTERN2) -> list[str]:
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

def plot_N_vs_tauR(N: np.ndarray, tauR: np.ndarray) -> io.BytesIO:
    """Plot tauR versus N on a log-log scale and save the figure to a BytesIO object."""
    plt.figure()
    plt.loglog(N, tauR, marker='o')
    plt.title('Residue length (N) vs. Relaxation time (tauR)')
    plt.xlabel('Polymer length (N)')
    plt.ylabel('Relaxation time (tauR)')
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img

def write_image_to_directory(img: io.BytesIO, directory: str, filename: str) -> None:
    """Write image data to a file in the specified directory_path."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        f.write(img.read())
    #end-of-with
#end-of-function

def compute_tauR(dirName): # this is incorrect
    # Read the data from cm.dat and r2.dat files
    r_squared = np.loadtxt(os.path.join(dirName, "r2.dat"))
    # Calculate the average end-to-end distance squared (R^2)
    average_r2 = np.mean(r_squared)
    # Plot R^2 values against time
    time = np.arange(len(r_squared))
    # Fit the data to an exponential decay model
    def exponential_decay(t, tau):
        return average_r2 * np.exp(-t/tau)
    popt, pcov = curve_fit(exponential_decay, time, r_squared)
    relaxation_time = popt[0]
    return relaxation_time

def process_dirs():
    directories = get_directories(SOURCE_PATH)
    tauRList = []
    residueLengthList = []
    for directory in directories:
        relaxation_time = compute_tauR(directory)
        tauRList.append(relaxation_time)
        _, _, _, lengthN = extract_values(directory)
        residueLengthList.append(lengthN)
    # end-of-for

    # Sort the res_length array
    sorted_indices = np.argsort(residueLengthList)
    residueLengthList = np.array(residueLengthList)[sorted_indices]
    tauRList = np.array(tauRList)[sorted_indices]

    img = plot_N_vs_tauR(residueLengthList, tauRList)
    write_image_to_directory(img, SOURCE_PATH, "relaxation_time_vs_N.py.png")


# Main program
if __name__ == '__main__':
    process_dirs()
