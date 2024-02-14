from typing import List
import numpy as np

def save_two_lists(lags: List[int], autocorrelation_values: List[float], file_name: str) -> None:
    data_to_save = np.column_stack((lags, autocorrelation_values))
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')