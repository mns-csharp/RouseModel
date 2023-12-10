# File: plot_tau_R.py
from scipy.optimize import curve_fit
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import re
import glob
import io


SOURCE_PATH = r'C:\git\rouse_data\mc010'
DEST_PATH = r'C:\git\rouse_data\mc010'
DAT_FILE = "cm.dat"
TAU_MAX = 100
TEXT_FILE_NAME = "curvature.txt"
HEADERS = 'tauMax, inner, outer, factor, res, curvature\n'
PATTERN = r'run\d+_inner(\d+)_outer(\d+)_factor(\d+)_residue(\d+)'


from scipy.optimize import least_squares



def compute_tauR(dirName):
    # Load your R^2 data from a file
    r_squared = np.loadtxt(os.path.join(dirName, "r2.dat"))
    # Compute the mean
    mean_r_squared = np.mean(r_squared)
    # Compute the autocorrelation function
    gR = np.correlate(r_squared - mean_r_squared, r_squared - mean_r_squared, mode='full')[-len(r_squared):]
    gR /= gR[0]

    # Define the initial guess for tauR
    initial_guess = 1.0

    def decay_func(params, t, gR):
        tauR = params[0]
        residuals = gR - np.exp(-t / tauR)
        return residuals

    # Fit the autocorrelation function to the decay function using nonlinear least squares
    result = least_squares(decay_func, initial_guess, args=(np.arange(len(gR)), gR))

    # The relaxation time tauR is the fitted parameter
    tauR = result.x[0]

    return tauR

def plot_N_vs_tauR(N: np.ndarray, tauR: np.ndarray, x_range: tuple[float, float]) -> io.BytesIO:
    """Plot tauR versus N on a log-log scale and save the figure to a BytesIO object."""
    plt.figure()
    plt.loglog(N, tauR, marker='o')
    plt.xlabel('Polymer length (N)')
    plt.ylabel('Relaxation time (tauR)')
    plt.title('Residue length (N) vs. Relaxation time (tauR)')
    plt.grid(True)
    plt.xlim(x_range[0], x_range[1])  # Set the x-axis range
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


def process_dirs(start_directory: str=SOURCE_PATH):
    dir_list = get_directories(start_directory)
    tauRList = []
    residueLengthList = []
    for dirName in dir_list:
        tauR = compute_tauR(dirName)
        tauRList.append(tauR)
        _, _, _, length = extract_values(dirName)
        residueLengthList.append(length)

    # Sort polymer lengths in ascending order
    sorted_indices = np.argsort(residueLengthList)
    sorted_lengths = np.array(residueLengthList)[sorted_indices]
    sorted_tauR = np.array(tauRList)[sorted_indices]

    # Set the desired x-axis range
    x_range = (0, 500)

    # Plot and save the sorted data
    img = plot_N_vs_tauR(sorted_lengths, sorted_tauR, x_range)
    write_image_to_directory(img=img, directory=DEST_PATH, filename="plot_tau_R.py.png")


if __name__ == "__main__":
    process_dirs()
