from utils.backend import np

from NN.layer import Layer


class ZeroPad(Layer):
    def __init__(self, left: int = 1, right: int = 1, top: int = 1, bottom: int = 1):
        super().__init__()
        self.pads = (left, right, top, bottom)

    def predict(self, X: np.ndarray) -> np.ndarray:
        N, K, H, W = X.shape
        # print(X.shape)
        left, right, top, bottom = self.pads
        zp = np.zeros((N, K, H + top + bottom, W + left + right))
#         print(zp.shape)
#         print(zp[:, :, top:H + bottom, left:W + right].shape)
        zp[:, :, top:H + top, left:W + left] = X
        return zp

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def backward(self, dL_dZ: np.ndarray, calc_grads: bool = True) -> np.ndarray:
        left, right, top, bottom = self.pads
        return dL_dZ[:, :, top: -bottom, left: -right]
