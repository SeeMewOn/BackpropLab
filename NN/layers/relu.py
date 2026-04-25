from utils.backend import np

from NN.layer import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.A: np.ndarray = np.array([])

    def predict(self, A: np.ndarray) -> np.ndarray:
        return np.maximum(0, A)

    def forward(self, A: np.ndarray) -> np.ndarray:
        self.A = A
        return self.predict(A)

    def backward(self, dL_dZ: np.ndarray, calc_grads: bool = True) -> np.ndarray:
        dZ_dA = (self.A > 0).astype(int)
        dL_dA = np.multiply(dZ_dA, dL_dZ)
        return dL_dA
