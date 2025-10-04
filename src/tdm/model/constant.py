import numpy as np


class ConstantProbabilityModel:
    def __init__(self, p: float) -> None:
        assert (p >= 0) and (p <= 1)
        self.p = p

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.repeat(self.p, repeats=features.shape[0])
