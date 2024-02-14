import numpy as np
import matplotlib
from framework.read_vec3 import read_vec3
matplotlib.use('Agg')
from typing import List, Tuple, Optional

def ComputeCForRange(vectors: np.ndarray, max_lag: int = 1000, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    n = len(vectors)
    # Ensure vectors is a 2D array
    if vectors.ndim == 1:
        vectors = vectors[:, np.newaxis]
    # Efficiently compute the autocorrelation using FFT for all lags at once
    fft_vectors = np.fft.fft(vectors, n=2*n, axis=0)  # Specify axis for FFT operation
    autocorrelations = np.fft.ifft(fft_vectors * np.conjugate(fft_vectors), axis=0).real
    autocorrelations = autocorrelations[:max_lag+1, :]  # Take only the needed lags
    # Sum across the rows to collapse into a 1D array
    autocorrelations = np.sum(autocorrelations, axis=1)
    # Correct the normalizations for the number of overlapping points
    overlap = np.arange(n, n-max_lag-1, -1)
    autocorrelations /= overlap
    # Normalize the autocorrelations to 1 at lag 0
    autocorrelations /= autocorrelations[0]
    # If threshold is specified, truncate the result where the autocorrelation falls below the threshold
    if threshold is not None:
        below_threshold_indices = np.where(autocorrelations < threshold)[0]
        if below_threshold_indices.size > 0:
            first_below_threshold = below_threshold_indices[0]
            autocorrelations = autocorrelations[:first_below_threshold]
            lags = np.arange(first_below_threshold)
        else:
            lags = np.arange(max_lag + 1)
    else:
        lags = np.arange(max_lag + 1)
    return lags, autocorrelations


if __name__ == "__main__":
    r_end_vec = r"framework\test_data_2.txt"
    dir1 = "."
    chain1Vec3List = read_vec3(dir_path=dir1, file_name=r_end_vec)
    print(chain1Vec3List)
    lags, autos = ComputeCForRange(chain1Vec3List, 3)
    print(lags)
    print(autos)

