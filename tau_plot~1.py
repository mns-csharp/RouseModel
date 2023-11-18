# plot until the maximum `tau_max`
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load positions from file, skipping the first row (header)
positions = np.loadtxt('cm.dat', skiprows=1)

# Generate time values based on the number of positions
# Assuming the time step between each position is 1
times = np.arange(len(positions))

# Define tau_max
tau_max = 100  # replace with your desired value

# Calculate MSD for each lag time
τ = np.arange(1, tau_max + 1)  # [1, 2, ..., tau_max]
MSD = np.empty_like(τ, dtype=float)

for i in range(1, tau_max + 1):
    displacements = positions[i:] - positions[:-i]  # calculate displacements for this lag time
    squared_displacements = np.sum(displacements**2, axis=1)  # calculate squared displacements
    MSD[i - 1] = np.mean(squared_displacements)  # calculate MSD for this lag time

# Plot MSD vs. τ on a log-log scale
plt.loglog(τ, MSD, marker='o')
plt.xlabel('Lag time τ')
plt.ylabel('MSD')
plt.title('MSD vs. Lag time')
plt.grid(True)
plt.savefig('msd_plot.png')