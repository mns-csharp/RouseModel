import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def plot_diffusion_vs_length(diffusion_coefficients: np.ndarray, lengths: np.ndarray) -> io.BytesIO:
    """Plot the diffusion coefficients against the polymer lengths on a log-log scale and return the plot as a BytesIO object."""
    plt.figure()
    plt.loglog(lengths, diffusion_coefficients, 'o')
    plt.xlabel('Polymer length N')
    plt.ylabel('Diffusion coefficient D')
    plt.title('Diffusion coefficient vs. Polymer length (log-log scale)')
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img

def plot_diffusion_vs_length___():
    # Generate synthetic data that follows the power law D ~ N^(-nu)
    N = np.logspace(1, 3, num=50)  # Polymer lengths
    nu = 3/5
    D = N ** -nu  # Diffusion coefficients

    img = plot_diffusion_vs_length(D, N)

    # Write the BytesIO object to a file
    with open("plot_diffusion_vs_length___test.png", "wb") as f:
        f.write(img.read())

if __name__ == "__main__":
    plot_diffusion_vs_length___()