from utils.backend import np

from NN.layer import Layer


class Dropout(Layer):
    def __init__(self, p=0.5, is_training=True):
        """
        Слой, для выключения нейронов дальнейших слоёв
        :param p: Вероятность того, что нейрон останется включенным.
        """
        super().__init__(is_training=is_training)
        self.mask: np.ndarray = np.array([])
        self.p = p

    def predict(self, X: np.ndarray) -> np.ndarray:
        # if self.is_training:
        #     mask = (np.random.rand(*X.shape) < self.p) / self.p
        #     return X * mask
        return X

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.is_training:
            self.mask = (np.random.rand(*X.shape) < self.p) / self.p
            return X * self.mask
        return X

    def backward(self, dL_dA: np.ndarray, calc_grads: bool = True) -> np.ndarray:
        if self.is_training:
            return dL_dA * self.mask
        return dL_dA
