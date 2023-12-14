# File: msd_vs_tau_plot.py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import re
import glob
import io
import datetime
import math


SOURCE_PATH = r'C:\git\rouse_data\mc009'
DEST_PATH = r'C:\git\rouse_data\mc009\msd~1'
DAT_FILE = "cm.dat"
TAU_MAX = 100
TEXT_FILE_NAME = "curvature.txt"
HEADERS = 'tauMax, inner, outer, factor, res, curvature\n'
PATTERN = r'run\d+_inner(\d+)_outer(\d+)_factor(\d+)_residue(\d+)'


def current_date_to_dir_name() -> str:
    now = datetime.datetime.now()
    dir_name = now.strftime('%Y-%m-%d_%H-%M-%S')
    return dir_name


def load_CM_positions(filename: str) -> list:
    with open(filename, 'r') as file:
        next(file)  # Skip the first row (header)
        positions = [list(map(float, line.strip().split())) for line in file]
    return positions


def get_lag_times(tau_max: int) -> list:
    return list(range(1, tau_max + 1))


def calculate_MSD(positions: list, tau_max: int=0) -> tuple:
    if tau_max != 0:
        tau_max = int(len(positions) / tau_max)
    else:
        tau_max = len(positions) - 1
    tau = get_lag_times(tau_max)
    MSD = []
    for i in range(1, tau_max + 1):
        displacements = [math.sqrt(sum((p2 - p1) ** 2 for p1, p2 in zip(positions[j], positions[j + i]))) for j in range(len(positions) - i)]
        MSD.append(sum(displacements) / len(displacements) if displacements else float('nan'))
    return tau, MSD


def plot_msd_vs_tau(tau: list, MSD: list) -> io.BytesIO:
    plt.figure()
    plt.loglog(tau, MSD, marker='o')
    plt.xlabel('Lag time N')
    plt.ylabel('tauR')
    plt.title('tauR vs. Lag time')
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img


def calculate_curvature(tau: list, MSD: list) -> list:
    if len(tau) < 4 or len(MSD) < 4:
        return []
    spl = UnivariateSpline(tau, MSD)
    spl_1d = spl.derivative(n=1)
    spl_2d = spl.derivative(n=2)
    curvature = [abs(spl_2d(t)) / (1 + spl_1d(t)**2)**(3/2) for t in tau]
    return [round(c, 3) for c in curvature]


def write_image_to_directory(img: io.BytesIO, directory: str, filename: str) -> None:
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        f.write(img.read())


def clear_text_file(filepath: str) -> None:
    if os.path.exists(filepath):
        with open(filepath, 'w') as file:
            file.write('')


def extract_values(input_string: str) -> tuple:
    try:
        match = re.search(PATTERN, input_string)
        if match:
            return tuple(int(match.group(i)) for i in range(1, 5))
        else:
            raise ValueError("Input string does not match required format.")
    except ValueError as e:
        print(f"Error with directory_path {input_string}: {str(e)}")


def write_curvature_to_textfile(directory: str, curvature_value: float, dest_path: str=DEST_PATH) -> None:
    os.makedirs(dest_path, exist_ok=True)
    text_file_path = os.path.join(dest_path, TEXT_FILE_NAME)
    if not os.path.exists(text_file_path) or os.path.getsize(text_file_path) == 0:
        with open(text_file_path, 'w') as f:
            f.write(HEADERS)
    directory_name = os.path.basename(directory)
    inner, outer, factor, res = extract_values(directory_name)
    line = f"{TAU_MAX}, {inner}, {outer}, {factor}, {res}, {curvature_value}\n"
    with open(text_file_path, 'a') as f:
        f.write(line)


def main():
    directories = glob.glob(os.path.join(SOURCE_PATH, 'run*'))
    for directory in directories:
        dat_file_path = os.path.join(directory, DAT_FILE)
        positions = load_CM_positions(dat_file_path)
        tau, MSD = calculate_MSD(positions, TAU_MAX)
        img = plot_msd_vs_tau(tau, MSD)
        curvature = calculate_curvature(tau, MSD)
        max_curvature = max(curvature, default=0)
        write_image_to_directory(img, directory, 'msd_vs_tau_plot_2.png')
        write_curvature_to_textfile(directory, max_curvature)

    dest_dir_name = current_date_to_dir_name()
    dest_directory = os.path.join(DEST_PATH, dest_dir_name)
    clear_text_file(os.path.join(dest_directory, TEXT_FILE_NAME))


if __name__ == "__main__":
    main()