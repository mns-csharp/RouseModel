# File: plot_r2.py
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

def autocorrelation(data, tau):
    n = len(data)
    mean = sum(data) / n
    num = 0.0
    denom = 0.0

    for i in range(n - tau):
        num += (data[i] - mean) * (data[i + tau] - mean)
        denom += (data[i] - mean) ** 2

    autocorr = num / denom
    return autocorr


def main():
    # Assuming you have already loaded your R values
    data = np.loadtxt("r2.dat")
    # Sample data
    tau_values = range(1001)  # Range of tau values from 0 to 1000

    autocorr_tau0 = autocorrelation(data, 0)  # Autocorrelation for tau=0

    autocorr_values = [autocorrelation(data, tau) / autocorr_tau0 for tau in tau_values]

    # Plot autocorrelation values
    plt.figure(figsize=(10, 6))
    plt.plot(tau_values, autocorr_values)
    plt.xlabel('Tau')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Plot')
    plt.grid(True)
    plt.savefig("autocorrelation_of_R2.png")


if __name__ == '__main__':
    main()