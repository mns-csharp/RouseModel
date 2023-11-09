import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = np.loadtxt('cm.dat')

# Create the x values (assumed to be the index of the data)
x = np.arange(len(data))

# Create separate y values for each column
y1 = data[:, 0]  # cm-X values
y2 = data[:, 1]  # cm-Y values
y3 = data[:, 2]  # cm-Z values

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(x, y1, label='cm-X')
plt.plot(x, y2, label='cm-Y')
plt.plot(x, y3, label='cm-Z')

plt.xlabel('Index (can be time or steps)')
plt.ylabel('Value')
plt.title('Plot of cm-X, cm-Y, cm-Z')
plt.legend()

plt.show()