from typing import List, Tuple
import warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

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