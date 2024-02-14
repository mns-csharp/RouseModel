import numpy as np
import os
import matplotlib

from framework.ComputeCForRange import ComputeCForRange
from framework.fit import fit
from framework.read_vec3 import read_vec3
from framework.save_one_list import save_one_list
from framework.save_two_lists import save_two_lists

matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    root_dir = r'/home/mohammad/bioshell_v4/BioShell/target/release/Sikorski_Figure_7/20240124_174800'
    #r"C:\git\rouse_data~~\20240124_174800---" #r"/home/mohammad/bioshell_v4/BioShell/target/release/Sikorski_Figure_7/20240115_053051/"
    #r"/home/mohammad/bioshell_v4/BioShell/target/release/Sikorski_Figure_7/20240124_174800/"
    r_end_vec = "r_end_vec.dat"

    dir1 = os.path.join(root_dir, "run00_inner100000_outer100_factor1_residue50")
    dir2 = os.path.join(root_dir, "run01_inner100000_outer100_factor1_residue100")
    dir3 = os.path.join(root_dir, "run02_inner100000_outer100_factor1_residue150")


    print(f'Reading started for : {dir1}\{r_end_vec}')
    chain1Vec3List = read_vec3(dir_path=dir1, file_name=r_end_vec)
    print(f'Reading done')
    print(f'Autocorr started for : {dir1}\{r_end_vec}')
    autocorr1Lags, autocorr1values = ComputeCForRange(chain1Vec3List)
    print(f'Autocorr done')
    print(f'Fitting started for : {dir1}\{r_end_vec}')
    autocorr1LagsList = list(autocorr1Lags.tolist())
    autocorr1valuesList = list(autocorr1values.tolist())
    x_dataList1, y_fitList1, minimizing_pointList1 = fit(autocorr1LagsList, autocorr1valuesList)
    print(f'Fitting done')

    print(f'Reading started for : {dir2}\{r_end_vec}')
    chain2Vec3List = read_vec3(dir_path=dir2, file_name=r_end_vec)
    print(f'Reading done')
    print(f'Autocorr started for : {dir2}\{r_end_vec}')
    autocorr2Lags, autocorr2values = ComputeCForRange(chain2Vec3List)
    print(f'Autocorr done')
    print(f'Fitting started for : {dir2}\{r_end_vec}')
    autocorr2LagsList = list(autocorr2Lags.tolist())
    autocorr2valuesList = list(autocorr2values.tolist())
    x_dataList2, y_fitList2, minimizing_pointList2 = fit(autocorr2LagsList, autocorr2valuesList)
    print(f'Fitting done')


    print(f'Reading started for : {dir3}\{r_end_vec}')
    chain3Vec3List = read_vec3(dir_path=dir3, file_name=r_end_vec)
    print(f'Reading done')
    print(f'Autocorr started for : {dir3}\{r_end_vec}')
    autocorr3Lags, autocorr3values = ComputeCForRange(chain3Vec3List)
    print(f'Autocorr done')
    print(f'Fitting started for : {dir3}\{r_end_vec}')
    autocorr3LagsList = list(autocorr3Lags.tolist())
    autocorr3valuesList = list(autocorr3values.tolist())
    x_dataList3, y_fitList3, minimizing_pointList3 = fit(autocorr3LagsList, autocorr3valuesList)
    print(f'Fitting done')

    save_two_lists(autocorr1Lags, autocorr1values, "autocorr1.txt")
    save_two_lists(autocorr2Lags, autocorr2values, "autocorr2.txt")
    save_two_lists(autocorr3Lags, autocorr3values, "autocorr3.txt")

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
