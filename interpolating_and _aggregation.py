import os
import re
import math
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np

# Generate the x and y values for each line
x_list_1 = np.array([0, 1, 2])
y_list_1 = np.array([5, 7, 6])

x_list_2 = np.array([0, 2, 4])
y_list_2 = np.array([8, 4, 8])

x_list_3 = np.array([6, 8, 10])
y_list_3 = np.array([3, 4, 3])

# Plot each of the individual lines
plt.plot(x_list_1, y_list_1, label='Line-1')
plt.plot(x_list_2, y_list_2, label='Line-2')
plt.plot(x_list_3, y_list_3, label='Line-3')

# To calculate the mean, we need to interpolate the y-values at common x-values.
# Assuming we want the mean line for the entire range, we can interpolate between existing points.
all_x = np.unique(np.concatenate((x_list_1, x_list_2, x_list_3)))
interp_y1 = np.interp(all_x, x_list_1, y_list_1, left=None, right=None)
interp_y2 = np.interp(all_x, x_list_2, y_list_2, left=None, right=None)
interp_y3 = np.interp(all_x, x_list_3, y_list_3, left=None, right=None)

# Since interp will return nan for values outside the range of the input x-array,
# we need to handle these before calculating the mean.
combined_y = np.array([interp_y1, interp_y2, interp_y3])
valid_y = np.nan_to_num(combined_y, nan=np.nanmean(combined_y)) # Replace nan with mean of non-nan values
mean_y = np.nanmean(valid_y, axis=0) # Calculate the mean, ignoring nan values

# Plot the mean line
plt.plot(all_x, mean_y, label='Mean Line', linestyle='--')

# Add legend and labels
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of Lines and their Mean')

# Show the plot
plt.show()