import numpy as np

def save_one_list(lags: np.ndarray, file_name: str) -> None:
    data_to_save = np.column_stack((lags))
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')