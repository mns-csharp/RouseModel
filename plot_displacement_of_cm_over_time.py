import numpy as np
import matplotlib.pyplot as plt
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Plot CM displacement over time.')
parser.add_argument('--start-time', type=int, help='The starting time step (default is the first row of the file)')
parser.add_argument('--end-time', type=int, help='The ending time step (default is the last row of the file)')

# Parse the command-line arguments
args = parser.parse_args()

# Load the data from the file, skipping the first row
data = np.loadtxt('CM.dat', skiprows=1)

# If start-time and/or end-time are specified, select the corresponding rows from the data
if args.start_time is not None:
    data = data[args.start_time:]
if args.end_time is not None:
    data = data[:args.end_time]

# Calculate the differences between consecutive rows
diff = np.diff(data, axis=0)

# Calculate the displacement
displacement = np.sqrt(np.sum(diff**2, axis=1))

# Create a new figure and an axes
fig, ax = plt.subplots()

# Plot the displacement over time
ax.plot(displacement)

# Set the labels
ax.set_xlabel('Time step')
ax.set_ylabel('Displacement')

# Show the plot
plt.show()