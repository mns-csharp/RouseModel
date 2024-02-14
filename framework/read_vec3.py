import numpy as np
import os

def read_vec3(dir_path: str, file_name: str) -> np.ndarray:
    file_path = os.path.join(dir_path, file_name)
    # Load the entire file into a NumPy array, skipping the header row
    try:
        # Use NumPy's genfromtxt to read the file, skipping the first row
        data = np.genfromtxt(file_path, dtype=float, skip_header=1, invalid_raise=False)
        # Check if the data is 2D and has 3 columns
        if data.ndim == 1 and data.size == 0:
            # Empty or invalid file, return empty array
            return np.empty((0, 3), dtype=float)
        elif data.ndim == 1:
            # Data was 1D, meaning there was only one valid line in the file
            return np.array([data])
        elif data.shape[1] != 3:
            # The file does not have 3 columns, raise an error
            raise ValueError("Data does not have 3 columns")
        return data
    except ValueError as e:
        # Handle cases where the file cannot be processed by NumPy
        print(f"Error reading file {file_path}: {e}")
        return np.empty((0, 3), dtype=float)

if __name__ == "__main__":
    vec3array = read_vec3(".", "test_data.txt")
    print(vec3array)