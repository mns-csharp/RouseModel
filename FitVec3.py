import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeWarning
import warnings
from typing import List, Tuple, Optional

def save_one_list(lags: np.ndarray, file_name: str) -> None:
    data_to_save = np.column_stack((lags))
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')

def save_two_lists(lags: List[int], autocorrelation_values: List[float], file_name: str) -> None:
    data_to_save = np.column_stack((lags, autocorrelation_values))
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')

def save_fit(list1: List[float], list2: List[float], list3: List[float], file_name: str) -> None:
    data_to_save = np.column_stack((list3))
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')
    with open(file_name, 'ab') as f:  # 'ab' mode for binary append
        np.savetxt(f, np.column_stack((list1, list2)), fmt='%f')

def parse_vec3(line: str) -> Optional[np.ndarray]:
    try:
        return np.fromiter(line.strip().split(), dtype=float)
    except ValueError:
        return None

def read_vec3(dir_path: str, file_name: str) -> np.ndarray:
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'r') as file:
        vec3_list = [parse_vec3(line) for line in file if line.strip() and parse_vec3(line) is not None]
    return np.array(vec3_list)

def fit(x_data: List[int], y_data: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("x_data len in fit() :", len(x_data))
    print("y_data len in fit() :", len(y_data))
    def model(x: np.ndarray, a: float, b: float) -> np.ndarray:
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
        return x_data, np.array([]), np.array(initial_guess)
    except RuntimeError as e:
        print("Error occurred during curve fitting:", e)
        return x_data, np.array([]), np.array(initial_guess)
    y_fit = model(x_data, *params)
    minimizing_point = params
    return (x_data, np.array(y_fit), np.array(minimizing_point))

def autocorrelation_list_normalized__1(vectors: np.ndarray, max_lag: int = 1000, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    n = len(vectors)
    autocorrelations = np.zeros(max_lag)
    # Calculate denominator (sum of the norms squared)
    denominator = np.sum(np.linalg.norm(vectors, axis=1) ** 2)
    # Calculate autocorrelation for each lag
    for t in range(max_lag):
        numerator = 0
        for i in range(n - t):
            numerator += np.dot(vectors[i], vectors[i + t])
        autocorrelations[t] = numerator / denominator
        print(f"lag={t}, autocorr={autocorrelations[t]}")
        # If a threshold is provided, stop when the first autocorrelation coefficient below it is encountered
        if threshold is not None and autocorrelations[t] < threshold:
            return autocorrelations[:t], np.arange(t)
    return np.arange(max_lag), autocorrelations

def autocorrelation_list_normalized__2(vectors: np.ndarray, max_lag: int = 1000, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    n = len(vectors)
    autocorrelations = np.zeros(max_lag)
    # Calculate denominator (sum of the norms squared)
    denominator = np.sum(np.linalg.norm(vectors, axis=1) ** 2)
    # Calculate autocorrelation for each lag using vectorized operations
    for t in range(max_lag):
        numerator = np.sum(vectors[:n-t] * vectors[t:n], axis=0)
        autocorrelations[t] = np.sum(numerator) / denominator
        print(f"lag={t}, autocorr={autocorrelations[t]}")
        # If a threshold is provided, stop when the first autocorrelation coefficient below it is encountered
        if threshold is not None and autocorrelations[t] < threshold:
            return autocorrelations[:t], np.arange(t)
    return np.arange(max_lag), autocorrelations

def autocorrelation_function(vectors: np.ndarray, t: int) -> float:
    N = vectors.shape[0] - t
    C_t = np.sum((vectors[:N]) * (vectors[t:N+t]), axis=0)
    return np.sum(C_t)

def autocorrelation_list_normalized(vectors: np.ndarray, max_lag: int = 1000, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    n = len(vectors)
    autocorrelations = np.zeros(max_lag)
    # Calculate denominator (sum of the norms squared)
    denominator = np.sum(np.linalg.norm(vectors, axis=1) ** 2)
    # Calculate autocorrelation for each lag using vectorized operations
    for t in range(max_lag):
        N = n - t  # Number of observations for each lag time
        numerator = np.sum(vectors[:N] * vectors[t:N+t], axis=0)
        autocorrelations[t] = np.sum(numerator) / (denominator / N)  # Normalize by N
        print(f"lag={t}, autocorr={autocorrelations[t]}, raw value={np.sum(numerator)}")
        # If a threshold is provided, stop when the first autocorrelation coefficient below it is encountered
        if threshold is not None and autocorrelations[t] < threshold:
            return autocorrelations[:t], np.arange(t)
    return np.arange(max_lag), autocorrelations

if __name__ == "__main__":
    root_dir = r'/home/mohammad/bioshell_v4/BioShell/target/release/Sikorski_Figure_7/20240124_174800'
    #r"C:\git\rouse_data~~\20240124_174800---" #r"/home/mohammad/bioshell_v4/BioShell/target/release/Sikorski_Figure_7/20240115_053051/"
    #r"/home/mohammad/bioshell_v4/BioShell/target/release/Sikorski_Figure_7/20240124_174800/"

    r_end_vec = "r_end_vec.dat"

    dir1 = os.path.join(root_dir, "run00_inner100000_outer100_factor1_residue50")
    dir2 = os.path.join(root_dir, "run01_inner100000_outer100_factor1_residue100")
    dir3 = os.path.join(root_dir, "run02_inner100000_outer100_factor1_residue150")

    print('data reading started')
    chain1Vec3List = read_vec3(dir_path=dir1, file_name=r_end_vec)
    print(f'Reading done : {dir1}\{r_end_vec}')
    chain2Vec3List = read_vec3(dir_path=dir2, file_name=r_end_vec)
    print(f'Reading done : {dir2}\{r_end_vec}')
    chain3Vec3List = read_vec3(dir_path=dir3, file_name=r_end_vec)
    print(f'Reading done : {dir3}\{r_end_vec}')

    print('Autocorr computation started')
    autocorr1Lags, autocorr1values = autocorrelation_list_normalized(chain1Vec3List)
    print(f'Autocorr computation done for : {dir1}\{r_end_vec}')
    autocorr2Lags, autocorr2values = autocorrelation_list_normalized(chain2Vec3List)
    print(f'Autocorr computation done for : {dir2}\{r_end_vec}')
    autocorr3Lags, autocorr3values = autocorrelation_list_normalized(chain3Vec3List)
    print(f'Autocorr computation done for : {dir3}\{r_end_vec}')


    save_two_lists(autocorr1Lags, autocorr1values, "autocorr1.txt")
    save_two_lists(autocorr2Lags, autocorr2values, "autocorr2.txt")
    save_two_lists(autocorr3Lags, autocorr3values, "autocorr3.txt")

    print('Fitting started')
    x_dataList1, y_fitList1, minimizing_pointList1 = fit(autocorr1Lags, autocorr1values)
    print(f'Autocorr computation done for : {dir1}\{r_end_vec}')
    x_dataList2, y_fitList2, minimizing_pointList2 = fit(autocorr2Lags, autocorr2values)
    print(f'Autocorr computation done for : {dir2}\{r_end_vec}')
    x_dataList3, y_fitList3, minimizing_pointList3 = fit(autocorr3Lags, autocorr3values)
    print(f'Autocorr computation done for : {dir3}\{r_end_vec}')

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












