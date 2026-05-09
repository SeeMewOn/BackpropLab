import time

from utils.backend import np

from NN.layer import Layer


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, shared_params:dict=None):
        """
        Для реализации Weight Tying необходимо, чтобы у первого слоя
        языковой модели (InputEmbedding) и предпоследнего её слоя (Dense)
        была общая матрица весов.

        Градиент целевой функции по этой матрице - это сумма производных
        функции потерь по W от InputEmbedding и Dense слоёв. Поэтому,
        помимо self.params у этих слоёв должен быть общий self.grads

        shared_params - это словарь {"params": params, "grads": grads}
        """
        super().__init__()
        # Training params
        if shared_params:
            self.params = shared_params["params"]
            self.grads = shared_params["grads"]
        else:
            W: np.ndarray = np.random.randn(output_size, input_size).astype(np.float32) * np.sqrt(2.0 / input_size)  # (M_l, M_(l-1))
            b: np.ndarray = np.zeros(output_size).astype(np.float32)  # (M_l,)
            self.params = [W, b]
            self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.X: np.ndarray = np.array([])

    def predict(self, X: np.ndarray) -> np.ndarray:
        W, b = self.params
        A = X @ W.T + b

        return A

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return self.predict(X)

    def backward(self, dL_dY: np.ndarray):
        W, b = self.params
        D_in = self.X.shape[-1]
        D_out = dL_dY.shape[-1]

        dL_dX = dL_dY @ W

        if self.is_training:
            # Сплющиваем (B, L) в одну ось, чтобы считать градиенты параметров
            # (B*L, D_out)
            dL_dY_flat = dL_dY.reshape(-1, D_out)
            # (B*L, D_in)
            X_flat = self.X.reshape(-1, D_in)

            dL_dW = dL_dY_flat.T @ X_flat  # (D_out, D_in)
            dL_db = np.sum(dL_dY_flat, axis=0) # (D_out,)

            # self.grads = [dL_dW, dL_db] (OLD)
            self.grads[0] += dL_dW
            self.grads[1] += dL_db

        return dL_dX

if __name__ == '__main__':

    B, L, d_in, d_out = 32, 1024, 512, 20000
    tensor = np.random.rand(B, L, d_in).astype(np.float32)
    fake_grad = np.random.rand(B, L, d_out).astype(np.float32)

    layer = Dense(512, 20000)

    # Forward test
    start = time.time()
    for t in range(5):
        layer.forward(tensor)
        print(f"\r{t}", end="")

    print()
    end = time.time()
    print(f"Forward time: {end - start}")

    # Backward test
    start = time.time()
    for t in range(5):
        layer.backward(fake_grad)
        # print(f"\r{t}", end="")
        print(t)
    
    print()
    end = time.time()
    print(f"Backward time: {end - start}")