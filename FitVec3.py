import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeWarning
import warnings

def save_one_list(lags, file_name):
    data_to_save = np.column_stack((lags))
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')

def save_two_lists(lags, autocorrelation_values, file_name):
    data_to_save = np.column_stack((lags, autocorrelation_values))
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')


def save_fit(list1, list2, list3, file_name):
    data_to_save = np.column_stack((list3))
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')
    with open(file_name, 'ab') as f:  # 'ab' mode for binary append
        np.savetxt(f, np.column_stack((list1, list2)), fmt='%f')

def parse_vec3(line):
    try:
        return np.fromiter(line.strip().split(), dtype=float)
    except ValueError:
        return None

def read_vec3(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'r') as file:
        vec3_list = [parse_vec3(line) for line in file if line.strip() and parse_vec3(line) is not None]
    return np.array(vec3_list)

def autocorrelation_function(vectors, t):
    N = vectors.shape[0] - t
    mean = np.mean(vectors[:N], axis=0)
    C_t = np.sum((vectors[:N] - mean) * (vectors[t:N+t] - mean), axis=0)
    return np.sum(C_t)

def autocorrelation_vec3(data, threshold_min=0.0001, threshold_max=0.9999, num_lag=1000):
    print("length of data", len(data))
    n = len(data)
    mean = np.mean(data, axis=0)
    # Pre-calculate squared differences from the mean for each data point
    diff = data - mean
    variance = np.mean(np.sum(diff**2, axis=1))
    # List to hold the results
    results = []
    # Function to calculate autocorrelation for a given lag
    def calc_autocorrelation(t):
        # Element-wise multiplication followed by sum over all elements
        autocorrelation = np.sum(diff[:n - t] * diff[t:], axis=0)
        autocorrelation = np.sum(autocorrelation)  # Sum over all dimensions
        autocorrelation /= ((n - t) * variance)
        return autocorrelation
    # Calculate autocorrelation for each lag
    for t in range(min(num_lag, n)):
        autocorrelation = calc_autocorrelation(t)
        # Only include autocorrelation if within the specified thresholds
        if threshold_min <= autocorrelation <= threshold_max:
            results.append((t, autocorrelation))
    # Sort the results by lag
    sorted_results = sorted(results, key=lambda x: x[0])
    # Extract lags and autocorrelation_values from the sorted results
    lags, autocorrelation_values = zip(*sorted_results) if sorted_results else ([], [])
    return list(lags), list(autocorrelation_values)

def autocorrelation_list_normalized(vectors, max_lag=1000, threshold=None):
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ValueError("Input 'vectors' must be a 2D NumPy array.")
    n_vectors, vector_length = vectors.shape
    # Pre-compute mean and normalization factor
    mean_vectors = np.mean(vectors, axis=0)
    vectors_centered = vectors - mean_vectors
    norm_factor = np.sum(vectors_centered ** 2)  # Summing over all elements to get a scalar
    # Initialize autocorrelation array
    autocorrelations = np.zeros(max_lag)
    # Compute autocorrelation for each lag
    for tau in range(max_lag):
        # Avoid computing the autocorrelation beyond the number of vectors
        if tau >= n_vectors:
            break
        product = np.sum(vectors_centered[:-tau if tau > 0 else None] * vectors_centered[tau:], axis=0)
        autocorrelations[tau] = np.sum(product)
    # Normalize the autocorrelation values
    autocorrelations /= norm_factor  # Now this is a scalar division
    # Apply threshold if given
    if threshold is not None:
        mask = autocorrelations >= threshold
        lags = np.arange(max_lag)[mask]
        autocorrelations = autocorrelations[mask]
    else:
        lags = np.arange(max_lag)
    return lags, autocorrelations

def fit(x_data, y_data):
    print("x_data len in fit() :", len(x_data))
    print("y_data len in fit() :", len(y_data))
    def model(x, a, b):
        return a * np.exp((-1)*b * x)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    initial_guess = [0.0, 0.0]
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
    return (x_data, np.array(y_fit), np.array(minimizing_point))


root_dir = r"/home/mohammad/bioshell_v4/BioShell/target/release/Sikorski_Figure_7/20240115_053051/"
#r"/home/mohammad/bioshell_v4/BioShell/target/release/Sikorski_Figure_7/20240124_174800/" 
##r"C:\git\rouse_data~~\20240124_174800---"
r_end_vec = "r_end_vec.dat"

dir1 = os.path.join(root_dir, "run03_inner1000_outer1000_factor10_residue132")
dir2 = os.path.join(root_dir, "run10_inner1000_outer1000_factor10_residue416")
dir3 = os.path.join(root_dir, "run17_inner1000_outer1000_factor10_residue701")

chain1Vec3List = read_vec3(dir_path=dir1, file_name=r_end_vec)
chain2Vec3List = read_vec3(dir_path=dir2, file_name=r_end_vec)
chain3Vec3List = read_vec3(dir_path=dir3, file_name=r_end_vec)


autocorr1Lags, autocorr1values = autocorrelation_vec3(chain1Vec3List)
autocorr2Lags, autocorr2values = autocorrelation_vec3(chain2Vec3List)
autocorr3Lags, autocorr3values = autocorrelation_vec3(chain3Vec3List)

save_two_lists(autocorr1Lags, autocorr1values, "autocorr1.txt")
save_two_lists(autocorr2Lags, autocorr2values, "autocorr2.txt")
save_two_lists(autocorr3Lags, autocorr3values, "autocorr3.txt")


x_dataList1, y_fitList1, minimizing_pointList1 = fit(autocorr1Lags, autocorr1values)
x_dataList2, y_fitList2, minimizing_pointList2 = fit(autocorr2Lags, autocorr2values)
x_dataList3, y_fitList3, minimizing_pointList3 = fit(autocorr3Lags, autocorr3values)

save_one_list(minimizing_pointList1, "minimizing_pointList1.txt")
save_one_list(minimizing_pointList2, "minimizing_pointList2.txt")
save_one_list(minimizing_pointList3, "minimizing_pointList3.txt")

save_two_lists(x_dataList1, y_fitList1, "fit1.txt")
save_two_lists(x_dataList2, y_fitList2, "fit2.txt")
save_two_lists(x_dataList3, y_fitList3, "fit3.txt")

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
plt.savefig('log_plot.png', dpi=300)












