import time

from NN.layer import Layer
from utils.backend import np


class LayerNorm(Layer):
	def __init__(self, d_model, eps=1e-6):
		super().__init__()
		# Trainable params
		g = np.ones(d_model).astype(np.float32)
		b = np.zeros(d_model).astype(np.float32)
		self.params = [g, b]

		# Hyperparams
		self.eps = eps

		# Cache
		self.cache = []

	def predict(self, X):
		return self.forward(X)

	def forward(self, X):
		g, b = self.params

		mean = np.mean(X, axis=-1, keepdims=True)  # (B, L, 1)
		X_shift = X - mean  # (B, L, D)
		var = np.mean(X_shift ** 2, axis=-1, keepdims=True)
		inv_std = 1.0 / np.sqrt(var + self.eps)  # (B, L, 1)
		X_norm = X_shift * inv_std  # (B, L, D)
		Y = g * X_norm + b

		# Добавляем данные в кэш для расчёта бэкпропа
		if self.is_training:
			self.cache = [X_norm, X_shift, inv_std]
		return Y

	def backward(self, dL_dY):
		g, b = self.params
		d_model = g.shape[0]
		X_norm, X_shift, inv_std = self.cache

		dL_dX_norm = dL_dY * g                                                                  # (B, L, D)
		dL_dvar = -0.5 * (inv_std ** 3) * np.sum(dL_dX_norm * X_shift, axis=-1, keepdims=True)  # (B, L, 1)
		d1 = dL_dX_norm * inv_std + (2.0 / d_model) * dL_dvar * X_shift                         # (B, L, D)
		dL_dmean = - np.sum(d1, axis=-1, keepdims=True)                                         # (B, L, 1)
		dL_dX = d1 + dL_dmean / d_model                                                         # (B, L, D)
		dL_dg = np.sum(dL_dY * X_norm, axis=(0, 1))                                             # (D,)
		dL_db = np.sum(dL_dY, axis=(0, 1))                                                      # (D,)
		self.grads = [dL_dg, dL_db]
		return dL_dX


if __name__ == '__main__':

	# (B, L, D)
	tensor = np.random.rand(32, 1024, 512)
	mha = LayerNorm(512)

	# Forward test
	start = time.time()
	for t in range(100):
		mha.forward(tensor)
		print(f"\r{t}", end="")

	print()
	end = time.time()
	print(f"Forward time: {end - start}")

	# Backward test
	start = time.time()
	for t in range(100):
		mha.backward(tensor)
		print(f"\r{t}", end="")

	print()
	end = time.time()
	print(f"Backward time: {end - start}")
