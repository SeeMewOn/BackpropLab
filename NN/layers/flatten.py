from utils.backend import np

from NN.layer import Layer


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        return X.reshape(N, -1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input_shape = X.shape  # Запоминаем (N, C, H, W)
        return self.predict(X)

    def backward(self, dL_dZ: np.ndarray, calc_grads: bool = True) -> np.ndarray:
        # dL_dZ пришел из Dense, он имеет форму (N, D)
        # Нам нужно вернуть его в форму (N, C, H, W) для Conv
        return dL_dZ.reshape(self.input_shape)
