# plot SD w.r.t. time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load positions from file, skipping the first row (header)
positions = np.loadtxt('cm.dat', skiprows=1)

# Generate time values based on the number of positions
# Assuming the time step between each position is 1
times = np.arange(len(positions))

# Calculate square displacement for each time
displacements = positions[1:] - positions[:-1]  # Calculate displacements for each time
squared_displacements = np.sum(displacements**2, axis=1)  # Calculate squared displacements

# Plot squared displacement vs. time
plt.plot(times[1:], squared_displacements, marker='o')
plt.xlabel('Time')
plt.ylabel('Squared Displacement')
plt.title('Squared Displacement vs. Time')
plt.grid(True)
plt.savefig('squared_displacement_plot.png')