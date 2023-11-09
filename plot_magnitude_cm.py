import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = np.loadtxt('cm.dat')

# Create the x values (assumed to be the index of the data)
x = np.arange(len(data))

# Calculate the magnitude of the center of mass
magnitude = np.sqrt(np.sum(data**2, axis=1))

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(x, magnitude, label='Magnitude of Center of Mass')

plt.xlabel('Time (or steps)')
plt.ylabel('Magnitude of Center of Mass')
plt.title('Plot of Magnitude of Center of Mass w.r.t. Time')
plt.legend()

plt.show()