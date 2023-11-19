import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import re
import glob
import io
import datetime

SOURCE_PATH = r'C:\git\rouse_data\mc005'
DEST_PATH = r'C:\git\rouse_data\mc005\msd~1'
DAT_FILE = "cm.dat"
TAU_MAX = 100
TEXT_FILE_NAME = "curvature.txt"
HEADERS = '1/tau_max, inner, outer, factor, res, curvature\n'

def current_date_to_dir_name() -> str:
    """Convert the current date and time into a directory name."""
    now = datetime.datetime.now()
    dir_name = now.strftime('%Y-%m-%d_%H-%M-%S')  # Format the datetime object as a string
    return dir_name

def load_positions(filename: str) -> np.ndarray:
    """Load positions from a file, skipping the first row (header)."""
    return np.loadtxt(filename, skiprows=1)


def get_lag_times(tau_max: int) -> np.ndarray:
    """Generate lag times based on tau_max."""
    return np.arange(1, tau_max + 1)


def calculate_msd(positions: np.ndarray, tau_max: int=0) -> tuple[np.ndarray, np.ndarray]:
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


def plot_msd(tau: np.ndarray, MSD: np.ndarray) -> io.BytesIO:
    """Plot MSD versus tau on a log-log scale and save the figure to a BytesIO object."""
    plt.figure()
    plt.loglog(tau, MSD, marker='o')
    plt.xlabel('Lag time tau')
    plt.ylabel('MSD')
    plt.title('MSD vs. Lag time')
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img


def calculate_curvature(tau: np.ndarray, MSD: np.ndarray) -> np.ndarray:
    """Calculate curvature of MSD using UnivariateSpline."""
    if len(tau) < 4 or len(MSD) < 4:
        return np.array([])
    spl = UnivariateSpline(tau, MSD)
    spl_1d = spl.derivative(n=1)
    spl_2d = spl.derivative(n=2)
    curvature = np.abs(spl_2d(tau)) / (1 + spl_1d(tau)**2)**(3/2)
    return np.round(curvature, 3)


def write_image_to_directory(img: io.BytesIO, directory: str, filename: str) -> None:
    """Write image data to a file in the specified directory."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        f.write(img.read())

def clear_text_file(filepath: str) -> None:
    """Clear the contents of a text file."""
    if os.path.exists(filepath):
        with open(filepath, 'w') as file:
            file.write('')

def extract_values(input_string: str) -> tuple[int, int, int, int]:
    """Extract values from a formatted string."""
    match = re.search(r'run\d+_inner(\d+)_outer(\d+)_factor(\d+)_residue(\d+)', input_string)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    else:
        raise ValueError("Input string does not match required format.")


def write_curvature_to_textfile(directory: str, curvature_value: float, dest_path: str=DEST_PATH) -> None:
    """Write headers to the curvature_values.txt file if it doesn't exist or is empty, then write directory name, filename, and curvature value to a text file."""
    # Create the directory if it doesn't exist
    os.makedirs(dest_path, exist_ok=True)
    text_file_path = os.path.join(dest_path, TEXT_FILE_NAME)

    # Write headers if the file doesn't exist or is empty
    if not os.path.exists(text_file_path) or os.path.getsize(text_file_path) == 0:
        with open(text_file_path, 'w') as f:
            f.write(HEADERS)

    # Obtain directory name
    directory_name = os.path.basename(directory)
    try:
        inner, outer, factor, res = extract_values(directory_name)
    except ValueError as e:
        print(f"Error with directory {directory_name}: {str(e)}")
        return

    # Prepare the line to be written
    line = f"{TAU_MAX}, {inner}, {outer}, {factor}, {res}, {curvature_value}\n"

    # Append the line to the text file
    with open(text_file_path, 'a') as f:
        f.write(line)


def plot_image_and_curvature(filename: str, tau_max: int=0) -> tuple[io.BytesIO, float]:
    """Generate a plot image of MSD and calculate the curvature."""
    positions = load_positions(filename)
    tau, MSD = calculate_msd(positions, tau_max)
    img = plot_msd(tau, MSD)
    curvature = calculate_curvature(tau, MSD)
    avg_curvature = np.nan if curvature.size == 0 else np.mean(curvature)
    return img, avg_curvature


def get_directories(dir_path: str, pattern: str='run*_inner*_outer*_factor*_res*') -> list[str]:
    """This function retrieves all directories in the specified path that match a given pattern."""
    return glob.glob(os.path.join(dir_path, pattern))


def process_directories(source_path: str=SOURCE_PATH, dest_path: str=DEST_PATH, dat_file: str=DAT_FILE) -> None:
    """Process all directories in source_path, calculate MSD and curvature, and write results to dest_path."""
    clear_text_file(os.path.join(dest_path, TEXT_FILE_NAME))
    directories = get_directories(source_path)
    for directory in directories:
        filename = os.path.join(directory, dat_file)
        if os.path.isfile(filename):
            img, curvature = plot_image_and_curvature(filename, tau_max=TAU_MAX)
            write_image_to_directory(img, dest_path, os.path.basename(directory) + '.png')
            write_curvature_to_textfile(directory, curvature, dest_path)
        else:
            print(f"File {filename} does not exist.")


if __name__ == "__main__":
    process_directories()