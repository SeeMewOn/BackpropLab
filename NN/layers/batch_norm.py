from NN.layer import Layer
from utils.backend import np


class BatchNorm(Layer):
    def __init__(self, d, eps=0.001, momentum=0.99, print_hyperparams=True):
        super().__init__()
        if print_hyperparams:
            print(f"BatchNorm:eps = {eps} momentum = {momentum}")

        # Trainable params
        gamma: np.ndarray = np.ones(d).astype(np.float32)
        beta: np.ndarray = np.zeros(d).astype(np.float32)
        self.params = [gamma, beta]

        # Hyperparams
        self.momentum = momentum
        self.eps = eps

        # Оценки матожидания и дисперсии для инференса
        inference_mean = np.zeros(d).astype(np.float32)
        inference_var = np.ones(d).astype(np.float32)
        self.state = [inference_mean, inference_var]

        # Buffer
        self.buffer = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        inference_mean, inference_var = self.state
        gamma, beta = self.params

        # Если BN стоит после Conv
        X_2d = X
        if X.ndim == 4:
            # (N, C, H, W) -> (C, N, H, W) -> (C, N * H * W) -> (N * H * W, C)
            N, C, H, W = X.shape
            X_2d = X.transpose(0, 2, 3, 1).reshape(-1, C)

        X_norm_2d = (X_2d - inference_mean) / np.sqrt(inference_var + self.eps)
        out_2d = gamma * X_norm_2d + beta

        # Если BN стоит после Conv
        if X.ndim == 4:
            # (N * H * W, C) -> (N, H, W, C) -> (N, C, H, W)
            N, C, H, W = X.shape
            out_2d = out_2d.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return out_2d

    def forward(self, X: np.ndarray) -> np.ndarray:
        gamma, beta = self.params

        # Если BN стоит после Conv
        X_2d = X
        if X.ndim == 4:
            # (N, C, H, W) -> (C, N, H, W) -> (C, N * H * W) -> (N * H * W, C)
            N, C, H, W = X.shape
            X_2d = X.transpose(0, 2, 3, 1).reshape(-1, C)

        # Вычисление среднего и дисперсии
        mean = np.mean(X_2d, axis=0)
        var = np.var(X_2d, axis=0)

        # Inference values update
        if self.is_training:
            inference_mean, inference_var = self.state
            self.state = [
                self.momentum * inference_mean + (1 - self.momentum) * mean,
                self.momentum * inference_var + (1 - self.momentum) * var,
            ]

        # X normalization
        X_norm_2d = (X_2d - mean) / np.sqrt(var + self.eps)

        # Buffer save X_norm, X - 2D
        self.buffer = [X_norm_2d, X_2d, mean, var]

        # Output 2d
        out_2d = gamma * X_norm_2d + beta

        # Если BN стоит после Conv
        if X.ndim == 4:
            # (N * H * W, C) -> (N, H, W, C) -> (N, C, H, W)
            N, C, H, W = X.shape
            out_2d = out_2d.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return out_2d

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        gamma, beta = self.params
        X_norm_2d, X_2d, mean, var = self.buffer
        m, _ = X_2d.shape

        # Если BN стоит после Conv
        dL_dout_2d = dL_dout
        if dL_dout.ndim == 4:
            # (N, C, H, W) -> (C, N, H, W) -> (C, N * H * W) -> (N * H * W, C)
            N, C, H, W = dL_dout.shape
            dL_dout_2d = dL_dout.transpose(0, 2, 3, 1).reshape(-1, C)

        # TODO Оптимизировать расчёты
        # Grads calc
        dL_dX_norm = dL_dout_2d * gamma
        v1 = var + self.eps
        d1 = dL_dX_norm / np.sqrt(v1)
        X1 = X_2d - mean
        dL_dvar = - 0.5 * np.power(v1, -3 / 2) * np.sum(dL_dX_norm * X1, axis=0)
        d2 = 2 / m * dL_dvar * X1
        dL_dmean = - np.sum(d1 + d2, axis=0)
        dL_dX_2d = d1 + d2 + dL_dmean / m

        if self.is_training:
            # Сохранение градиентов
            dL_dgamma = np.sum(dL_dout_2d * X_norm_2d, axis=0)
            dL_dbeta = np.sum(dL_dout_2d, axis=0)
            self.grads = [dL_dgamma, dL_dbeta]

        # Если BN стоит после Conv
        if dL_dout.ndim == 4:
            # (N * H * W, C) -> (N, H, W, C) -> (N, C, H, W)
            N, C, H, W = dL_dout.shape
            dL_dX_2d = dL_dX_2d.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return dL_dX_2d
