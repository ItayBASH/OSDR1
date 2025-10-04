import numpy as np


def truncate_to_valid_probabilities(predictions: np.ndarray, eps=1e-5) -> np.ndarray:
    p = predictions
    p = np.where(p > 1, 1 - eps, p)
    p = np.where(p < 0, 0 + eps, p)
    return p
