from utils.backend import np

from NN.layer import Layer


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # Training params
        W: np.ndarray = np.random.randn(output_size, input_size).astype(np.float32) * np.sqrt(2 / input_size)  # (M_l, M_(l-1))
        b: np.ndarray = np.zeros(output_size).astype(np.float32)  # (M_l,)
        self.params = [W, b]
        self.Z_prev: np.ndarray = np.array([])

    def predict(self, X: np.ndarray) -> np.ndarray:
        W, b = self.params
        A = X @ W.T + b

        return A

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z_prev = X
        return self.predict(X)

    def backward(self, dL_dA: np.ndarray):
        W, b = self.params

        dL_dZ_prev = dL_dA @ W

        if self.is_training:
            N = dL_dA.shape[0]  # Batch size
            dL_db = dL_dA.mean(axis=0)
            dL_dW = dL_dA.T @ self.Z_prev / N
            self.grads = [dL_dW, dL_db]

        return dL_dZ_prev
