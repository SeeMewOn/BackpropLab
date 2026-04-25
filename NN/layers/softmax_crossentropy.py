from utils.backend import np

from NN.layer import Layer


class SoftmaxCrossEntropy(Layer):
    def __init__(self):
        super().__init__()
        self.Z: np.ndarray = np.array([])

    def predict(self, A: np.ndarray) -> np.ndarray:
        A_exp = np.exp(A - np.max(A, axis=1, keepdims=True))
        S = np.sum(A_exp, axis=1, keepdims=True)
        Z = A_exp / S
        return Z

    def forward(self, A: np.ndarray) -> np.ndarray:
        Z = self.predict(A)
        self.Z = Z
        return Z

    def backward(self, T: np.ndarray, calc_grads: bool = True) -> np.ndarray:
        dL_dA = self.Z - T
        return dL_dA
