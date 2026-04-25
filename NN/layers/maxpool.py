from utils.backend import np
from utils.backend import as_strided

from NN.layer import Layer


class MaxPool(Layer):
    def __init__(self, n: int = 2, s: int = 2):
        super().__init__()
        self.s = s
        self.n = n
        self.X: np.ndarray = np.array([])
        self.mask: np.ndarray = np.array([])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _max_pool(X, self.n, self.s)[0]

    def forward(self, X: np.ndarray) -> np.ndarray:
        out, mask = _max_pool(X, self.n, self.s)
        self.X = X
        self.mask = mask
        return out

    def backward(self, dL_dA: np.ndarray, calc_grads: bool = True) -> np.ndarray:
        # TODO задокументировать этот ужас
        n = self.n
        N, C, H_out, W_out = dL_dA.shape
        dL_dA_windows = self.mask * dL_dA[..., None, None]
        dL_dZ = dL_dA_windows.transpose(0, 1, 2, 4, 3, 5)
        dL_dZ = dL_dZ.reshape((N, C, H_out * n, W_out * n))
        return dL_dZ


def _max_pool(X: np.ndarray, n: int = 2, s: int = 2):
    N, C, H, W = X.shape
    s_N, s_C, s_H, s_W = X.strides

    H_out = (H - n) // s + 1
    W_out = (W - n) // s + 1

    new_shape = (N, C, H_out, W_out, n, n)
    new_strides = (s_N, s_C, s * s_H, s * s_W, s_H, s_W)

    X_windows = as_strided(X, shape=new_shape, strides=new_strides)

    out = np.max(X_windows, axis=(4, 5))
    # Маска с максимальными значениями
    mask = X_windows == out[..., None, None]
    return out, mask
