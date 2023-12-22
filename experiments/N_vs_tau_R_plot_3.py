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


def loadR2(fileName: str) -> np.ndarray:
    """Load R^2 values from a file, skipping the first row (header)."""
    return np.loadtxt(fileName, skiprows=1)


def getLagTimes(tauMax: int) -> np.ndarray:
    """Generate lag times based on tauMax."""
    return np.arange(1, tauMax + 1)


def calculateMSD(positions: np.ndarray, tauMax: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Mean Square Displacement (tauR) for each lag time."""
    if tauMax != 0:
        tauMax = int(len(positions) / tauMax)
    else:
        tauMax = len(positions) - 1
    tau = getLagTimes(tauMax)
    MSD = np.empty_like(tau, dtype=float)
    for i in range(1, tauMax + 1):
        displacements = positions[i:] - positions[:-i]
        squaredDisplacements = np.sum(displacements**2, axis=1)
        MSD[i - 1] = np.mean(squaredDisplacements) if squaredDisplacements.size > 0 else np.nan
    return tau, MSD


from scipy.stats import linregress
def calculateMsdSlope(tau: np.ndarray, MSD: np.ndarray) -> float:
    """Calculate slope of the tauR-vs-N curve using linear regression."""
    slope, _, _, _, _ = linregress(x=MSD, y=tau)
    return slope


def calculateDiffCoeff(tau: np.ndarray, MSD: np.ndarray, d: int) -> float:
    """Calculate the diffusion coefficient from the tauR and delay time."""
    slope: float = calculateMsdSlope(tau, MSD)
    D = slope / (2 * d)
    return D


def calculateTauR(R2: np.ndarray, D: float) -> np.ndarray:
    """Calculate the Rouse time (tau_R) from the size of the polymer chain (R) and the diffusion coefficient (D)."""
    # check if D is not zero to avoid division by zero
    if D != 0:
        tau_R = R2 / D
    else:
        tau_R = None  # or any other value you deem appropriate when D is zero
    return tau_R


def plotNvsTauR(N: np.ndarray, tau_R: np.ndarray) -> io.BytesIO:
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


def getDirectories(dir_path: str, pattern: str= 'run*_inner*_outer*_factor*_res*') -> list[str]:
    """This function retrieves all directories in the specified path that match a given pattern."""
    return glob.glob(os.path.join(dir_path, pattern))


def extractValues(directoryName: str) -> tuple[int, int, int, int]:
    """Extract values from a formatted string."""
    try:
        match = re.search(r'run\d+_inner(\d+)_outer(\d+)_factor(\d+)_residue(\d+)', directoryName)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        else:
            raise ValueError("Input string does not match required format.")
        #end-of-if-else
    except ValueError as e:
        print(f"Error with mc_directory_path {directoryName}: {str(e)}")
    #end-of-try-except
#end-of-function


def writeImage(img: io.BytesIO, directory: str, filename: str) -> None:
    """Write image data to a file in the specified mc_directory_path."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        f.write(img.read())
    #end-of-with
#end-of-function


def load_CM_positions(filename: str) -> np.ndarray:
    """Load positions from a file, skipping the first row (header)."""
    return np.loadtxt(filename, skiprows=1)


def processDir(directory_path: str, cm_dat_file_name: str, r2_dat_file: str, time_interval: float) -> list:
    """Process a single mc_directory_path and return a list of [N, tau_R] or None if files do not exist."""
    cm_dat_path = os.path.join(directory_path, cm_dat_file_name)
    r2_dat_path = os.path.join(directory_path, r2_dat_file)
    if os.path.isfile(cm_dat_path) and os.path.isfile(r2_dat_path):
        positions = load_CM_positions(cm_dat_path)
        tau, MSD = calculateMSD(positions=positions, tauMax=TAU_MAX) # np.ndarray, np.ndarray
        D = calculateDiffCoeff(tau=tau, MSD=MSD, d=3) # float
        R2 = loadR2(r2_dat_path) # ndarray
        tau_R = calculateTauR(R2, D) # ndarray
        N = len(positions)
        return [N, np.mean(tau_R)]
    else:
        print(f"File {cm_dat_path} or {r2_dat_path} does not exist.")
        return None


def processDirectories(sourcePath: str=SOURCE_PATH, dest_path: str=DEST_PATH, cmDatFileName: str=DAT_FILE, r2DatFileName: str=R2_DAT_FILE, time_interval: float = 1.0) -> None:
    """Process all directories in sourcePath, calculate tau_R for different N, and write results to dest_path."""
    directoriesList = getDirectories(sourcePath)
    nVsTauList = []
    for dirItem in directoriesList:
        result = processDir(dirItem, cmDatFileName, r2DatFileName, time_interval)
        if result is not None:
            nVsTauList.append(result)
    nVsTauList.sort(key=lambda x: x[0])
    first_column, second_column = zip(*nVsTauList)
    img = plotNvsTauR(N=np.array(first_column), tau_R=np.array(second_column))
    writeImage(img, dest_path, 'N_vs_tau_R_plot.png')


if __name__ == "__main__":
    processDirectories()


