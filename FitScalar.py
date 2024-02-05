import numpy as np
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which is non-interactive and does not require Qt
import matplotlib.pyplot as plt


from scipy.optimize import OptimizeWarning
import warnings

import numpy as np

def save_one_list(lags, file_name):
    """
    Saves the lags and autocorrelation values to a text file.
    
    :param lags: List of lag values
    :param autocorrelation_values: List of autocorrelation values
    :param file_name: Name of the file to save the data to
    """
    # Convert the lags and autocorrelation_values to a NumPy array
    data_to_save = np.column_stack((lags))

    # Save the array to a text file
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')

def save_two_lists(lags, autocorrelation_values, file_name):
    """
    Saves the lags and autocorrelation values to a text file.
    
    :param lags: List of lag values
    :param autocorrelation_values: List of autocorrelation values
    :param file_name: Name of the file to save the data to
    """
    # Convert the lags and autocorrelation_values to a NumPy array
    data_to_save = np.column_stack((lags, autocorrelation_values))

    # Save the array to a text file
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')


def save_fit(list1, list2, list3, file_name):
    # Combine list1 and list2 into two columns and save
    data_to_save = np.column_stack((list3))
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')
    # Save list3 in a new line or as a new column, depending on your needs
    # This example code appends, assuming list3 is of the same length as list1 and list2
    with open(file_name, 'ab') as f:  # 'ab' mode for binary append
        np.savetxt(f, np.column_stack((list1, list2)), fmt='%f')


import numpy as np
import numpy as np
import os

def read_one_column(input_path, dat_file_name):
    file_path = os.path.join(input_path, dat_file_name)
    
    # Use NumPy's genfromtxt to read the file, assuming it's a text file with one numeric value per line
    data = np.genfromtxt(file_path, delimiter='\n', dtype=float, invalid_raise=False, comments=None)
    
    # Filter out non-numeric values that cannot be converted to float
    data = data[~np.isnan(data)]
    
    return data.tolist()


def auto_correlation_points_r2(data, threshold_min=0.0001, threshold_max=0.9999, num_lag=1000):
    n = len(data)
    autocorrelation_values = []
    lags = []
    
    # Compute the mean of the data
    mean = np.mean(data)

    # Compute the variance of the data
    variance = np.var(data)

    if num_lag > n:
        num_lag = n  # Ensure the number of lags does not exceed the data count.

    # Subtract mean and prepare data for autocorrelation calculation
    centered_data = np.array(data) - mean
    autocorrelations = np.zeros(num_lag)

    for t in range(num_lag):
        if t == 0:
            # For lag 0, the sum of products is simply the sum of squared centered data.
            sum_product = np.sum(centered_data ** 2)
        else:
            # For other lags, calculate the sum of products for the current lag.
            sum_product = np.sum(centered_data[:-t] * centered_data[t:])
        
        autocorr_value = sum_product / (n - t)

        # Normalize by variance if required
        if variance > 0:
            autocorr_value /= variance

        autocorrelations[t] = autocorr_value

    for t in range(num_lag):
        autocorr_value = autocorrelations[t]
        if threshold_min <= autocorr_value <= threshold_max:
            autocorrelation_values.append(autocorr_value)
            lags.append(t)

    return lags, autocorrelation_values

# Sample usage:
# Assuming data is already defined
# lags, autocorrelation_values = auto_correlation_points_r2(data)


def fit(x_data, y_data):
    def model(x, a, b):
        return a * np.exp((-1)*b * x)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    initial_guess = [1.0, 0.1]
    print("x_data len in fit() :", len(x_data))
    print("y_data len in fit() :", len(y_data))
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            params, covariance = curve_fit(model, x_data, y_data, p0=initial_guess)
            print("Fitted params:", params)
    except OptimizeWarning:
        print("Optimization warning occurred while fitting the curve.")
        return x_data, [], initial_guess
    except RuntimeError as e:
        print("Error occurred during curve fitting:", e)
        return x_data, [], initial_guess

    y_fit = model(x_data, *params)
    minimizing_point = params
    return x_data, y_fit.tolist(), minimizing_point.tolist()


root_dir = r"/home/mohammad/bioshell_v4/BioShell/target/release/Sikorski_Figure_7/20240124_174800/" 
r_end_vec = "r2.dat"

dir1 = os.path.join(root_dir, "run00_inner100000_outer100_factor1_residue50")
dir2 = os.path.join(root_dir, "run01_inner100000_outer100_factor1_residue100")
dir3 = os.path.join(root_dir, "run02_inner100000_outer100_factor1_residue150")

chain1ScalarList = read_one_column(dir_path=dir1, file_name=r_end_vec)
chain2ScalarList = read_one_column(dir_path=dir2, file_name=r_end_vec)
chain3ScalarList = read_one_column(dir_path=dir3, file_name=r_end_vec)

autocorr1Lags, autocorr1values = auto_correlation_points_r2(chain1ScalarList)
autocorr2Lags, autocorr2values = auto_correlation_points_r2(chain2ScalarList)
autocorr3Lags, autocorr3values = auto_correlation_points_r2(chain3ScalarList)

save_two_lists(autocorr1Lags, autocorr1values, "r2_autocorr1.txt")
save_two_lists(autocorr2Lags, autocorr2values, "r2_autocorr2.txt")
save_two_lists(autocorr3Lags, autocorr3values, "r2_autocorr3.txt")


x_dataList1, y_fitList1, minimizing_pointList1 = fit(autocorr1Lags, autocorr1values)
x_dataList2, y_fitList2, minimizing_pointList2 = fit(autocorr2Lags, autocorr2values)
x_dataList3, y_fitList3, minimizing_pointList3 = fit(autocorr3Lags, autocorr3values)

save_one_list(minimizing_pointList1, "r2_minimizing_pointList1.txt")
save_one_list(minimizing_pointList2, "r2_minimizing_pointList2.txt")
save_one_list(minimizing_pointList3, "r2_minimizing_pointList3.txt")

save_two_lists(x_dataList1, y_fitList1, "r2_fit1.txt")
save_two_lists(x_dataList2, y_fitList2, "r2_fit2.txt")
save_two_lists(x_dataList3, y_fitList3, "r2_fit3.txt")

xxx = [minimizing_pointList1[0], minimizing_pointList2[0], minimizing_pointList3[0]]
yyy = [minimizing_pointList1[1], minimizing_pointList2[1], minimizing_pointList3[1]]

inverted_list = [1/y if y != 0 else 0 for y in yyy]  # Inverting each element

print("x length", len(xxx), xxx)
print("y length", len(inverted_list), inverted_list)

# Convert list to numpy array for element-wise operations
xxx = np.array(xxx)  
inverted_list = np.array(inverted_list)

# Since the plot is log-log, the line will be plotted as log(y) = m*log(x) + log(c)
# Choose a point to define the line (x1, y1) and calculate c
x1 = xxx[0]
y1 = inverted_list[0]
c = y1 / (x1 ** 2.2)

# Generate x values for the line
line_x = np.linspace(min(xxx), max(xxx), 100)  # 100 points for a smooth line
# Calculate the corresponding y values for the line
line_y = c * line_x ** 2.2

# Create the log-log plot
plt.plot(xxx, inverted_list, 'o', label='Data Points')  # Plot the original points as a scatter plot
plt.plot(line_x, line_y, label='y = 2.2x')  # Plot the straight line

plt.xscale('log')
plt.yscale('log')

# Set the title and labels
plt.title('Log-Log Plot of Minimizing Points')
plt.xlabel('X values (log scale)')
plt.ylabel('Y values (log scale)')

# Add a legend to the plot
plt.legend()

# Save the plot with a logarithmic scale
plt.savefig('r2_log_plot.png', dpi=300)












