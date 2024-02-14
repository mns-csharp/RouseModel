from typing import List
import numpy as np

def save_fit(list1: List[float], list2: List[float], list3: List[float], file_name: str) -> None:
    data_to_save = np.column_stack((list3))
    np.savetxt(file_name, data_to_save, fmt='%f', comments='')
    with open(file_name, 'ab') as f:  # 'ab' mode for binary append
        np.savetxt(f, np.column_stack((list1, list2)), fmt='%f')