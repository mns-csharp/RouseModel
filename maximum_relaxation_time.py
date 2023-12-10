import glob
import io
import os

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


def compute_relaxation_time(r2_file):
    # Read the data from cm.dat and r2.dat files
    r2_data = np.loadtxt(r2_file, skiprows=1, usecols=0)
    # Calculate the average end-to-end distance squared (R^2)
    average_r2 = np.mean(r2_data)
    # Plot R^2 values against time
    time = np.arange(len(r2_data))
    # Fit the data to an exponential decay model
    def exponential_decay(t, tau):
        return average_r2 * np.exp(-t/tau)
    popt, pcov = curve_fit(exponential_decay, time, r2_data)
    relaxation_time = popt[0]
    return relaxation_time

def process_dirs():
    directories = get_directories(SOURCE_PATH)
    relaxation_times = []
    for directory in directories:
        r2_file = os.path.join(directory, R2_FILE)
        if os.path.isfile(r2_file):
            relaxation_time = compute_relaxation_time(r2_file)
            relaxation_times.append(relaxation_time)
        else:
            print(f"File {r2_file} does not exist.")
        #end-of-if-else
    #end-of-for
    max_relaxation_time = max(relaxation_times)
    print("Maximum Relaxation Time:", max_relaxation_time)
#end-of-function


# Main program
if __name__ == '__main__':
    process_dirs()

'''    
Output:
Maximum Relaxation Time: 182519897143.5263
'''