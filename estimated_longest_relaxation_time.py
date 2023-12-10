import numpy as np
from scipy.optimize import curve_fit


# Load your R^2 data from a file
r_squared = np.loadtxt('r2.dat')

# Your R^2 data
### r_squared = np.array([3212.136, 2964.903, 3029.043, 2280.498, 3552.020, 4967.750, 6473.641, 4393.817, 4442.758])

# Compute the mean
mean_r_squared = np.mean(r_squared)

# Compute the autocorrelation function
gR = np.correlate(r_squared - mean_r_squared, r_squared - mean_r_squared, mode='full')[-len(r_squared):]
gR /= gR[0]

# Define the exponential decay function
def decay_func(t, tauR):
    return np.exp(-t/tauR)

# Fit the autocorrelation function to the decay function
popt, pcov = curve_fit(decay_func, np.arange(len(gR)), gR)

# The relaxation time tauR is the fitted parameter
tauR = popt[0]
print(f"Estimated longest relaxation time (tauR): {tauR}")