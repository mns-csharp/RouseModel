import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

def plot_log_tauR_vs_log_N(tau_R: np.ndarray, N: np.ndarray) -> io.BytesIO:
    """Plot log(tau_R) against log(N) and return the plot as a BytesIO object."""
    fig, ax = plt.subplots()
    ax.loglog(N, tau_R, marker='o')
    ax.set_xlabel('N')
    ax.set_ylabel('tau_R')
    ax.set_title('tau_R vs. N in Logarithmic Scale')
    ax.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img

# Test the function
N = np.logspace(1, 7, num=50)
tau_R = np.logspace(1, 7, num=50)

img = plot_log_tauR_vs_log_N(tau_R, N)

# Write the BytesIO object to a file
with open("test_plot.png", "wb") as f:
    f.write(img.read())

