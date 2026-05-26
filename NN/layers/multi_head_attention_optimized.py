import time

from NN.functions import softmax
from utils.backend import np

from NN.layer import Layer

class MultiHeadSelfAttention(Layer):
	def __init__(self, d_model, n_head):
		super().__init__()
		# Trainable params
		# Инициализируем сразу целиком
		# Объединяем W_Q, W_K, W_V в одну матрицу (D, 3 * D)
		limit = np.sqrt(6 / (d_model + d_model))
		W_QKV = np.random.uniform(-limit, limit, (d_model, 3 * d_model)).astype(np.float32)
		W_O = np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)

		self.params = [W_QKV, W_O]
		self.grads = [np.zeros_like(W_QKV), np.zeros_like(W_O)]

		# Hyperparams
		# self.d_model = d_model
		self.n_head = n_head
		self.d_head = int(d_model / n_head)

		# Cache
		self.cache = []

	def predict(self, X, mask=None):
		return self.forward(X, mask)

	def forward(self, X, mask=None):
		"""
		X: (batch_size, sequence_length, d_model) или (B, L, D)
		"""
		B, L, D = X.shape
		W_QKV, W_O = self.params

		# (B, L, D) @ (D, D) -> (B, L, D)
		# Q = X @ W_Q
		# K = X @ W_K
		# V = X @ W_V

		# Один большой проход: (B, L, D) @ (D, 3*D) -> (B, L, 3*D)
		qkv = X @ W_QKV

		# Режем на Q, K, V и головы одновременно
		# (B, L, 3*D) -> (B, L, 3, H, d_h)
		qkv = qkv.reshape(B, L, 3, self.n_head, self.d_head)


		# Переставляем оси так, чтобы 3 (QKV) была на первом месте после батча
		# (B, L, 3, H, d_h) -> (3, B, H, L, d_h)
		qkv = qkv.transpose(2, 0, 3, 1, 4)
		Q, K, V = qkv[0], qkv[1], qkv[2]

		# Подаём полученные тензоры в SDPA. (B, H, L, d_h) -> (B, H, L, d_h)
		SDPA, SM = scaled_dot_product_attention(Q, K, V, mask)

		# (B, H, L, d_h) -> (B, L, H, d_h) -> (B, L, D)
		SDPA = SDPA.transpose(0, 2, 1, 3).reshape(B, L, D)




		# (B, L, D) ->  (B, L, D)
		Y = SDPA @ W_O

		# Добавляем данные в кэш для расчёта бэкпропа
		if self.is_training:
			self.cache = [X, Q, K, V, SM, SDPA]
		return Y

	def backward(self, dL_dY):
		W_QKV, W_O = self.params
		X, Q, K, V, SM, SDPA_3d = self.cache
		B, L, D = X.shape

		# 1. dL_dW_O
		O_flat = SDPA_3d.reshape(-1, D)  # (B, L, D) -> (B*L, D)
		dL_dY_flat = dL_dY.reshape(-1, D)  # (B, L, D) -> (B*L, D)
		dL_dW_O = O_flat.T @ dL_dY_flat  # (D, B*L) @ (B*L, D) -> (D, D)

		# 2. dL_dO
		dL_dO = dL_dY @ W_O.T  # (B, L, D) @ (D, D) -> (B, L, D)
		dL_dO = dL_dO.reshape(B, L, self.n_head, self.d_head)  # (B, L, D) -> (B, L, H, d_h)
		dL_dO = dL_dO.transpose(0, 2, 1, 3)  # (B, L, H, d_h) -> (B, H, L, d_h)

		# 3. dL_dSM (градиент по софтмаксу)
		dL_dSM = dL_dO @ V.swapaxes(-1, -2)  # (B, H, L, d_h) @ (B, H, d_h, L) -> (B, H, L, L)

		# 4. dL_dARG (градиент по аргументу софтмакса)
		dL_dARG = (dL_dSM - (dL_dSM * SM).sum(axis=-1, keepdims=True)) * SM  # (B, H, L, L)
		dL_dARG /= np.sqrt(self.d_head)

		# 5. dL_dQ, dL_dK, dL_dV
		# (B, H, L, L) @ (B, H, L, d_h) -> (B, H, L, d_h)
		dL_dQ = dL_dARG @ K
		dL_dK = dL_dARG.swapaxes(-1, -2) @ Q
		dL_dV = SM.swapaxes(-1, -2) @ dL_dO
		# (B, H, L, d_h) -> (B, L, H, d_h)
		dL_dQ = dL_dQ.transpose(0, 2, 1, 3)
		dL_dK = dL_dK.transpose(0, 2, 1, 3)
		dL_dV = dL_dV.transpose(0, 2, 1, 3)
		# (B, L, H, d_h) -> (B, L, D)
		dL_dQ = dL_dQ.reshape(B, L, D)
		dL_dK = dL_dK.reshape(B, L, D)
		dL_dV = dL_dV.reshape(B, L, D)

		# # 6. dL_dX
		# # (B, L, D) @ (D, D) + (B, L, D) @ (D, D) + (B, L, D) @ (D, D) = (B, L, D)
		# dL_dX = dL_dQ @ W_Q.T + dL_dK @ W_K.T + dL_dV @ W_V.T

		# 1. Склеиваем градиенты по Q, K, V в один тензор (B, L, 3*D)
		# np.concatenate — довольно быстрая операция
		dL_dQKV_3d = np.concatenate([dL_dQ, dL_dK, dL_dV], axis=-1)

		# # 7. dL_dW_Q, dL_dW_K, dL_dW_V
		# # (B, L, D) -> (B*L, D)
		# X = X.reshape(-1, D)
		# dL_dQ = dL_dQ.reshape(-1, D)
		# dL_dK = dL_dK.reshape(-1, D)
		# dL_dV = dL_dV.reshape(-1, D)
		# # (D, B*L) @ (B*L, D) -> (D, D)
		# dL_dW_Q = X.T @ dL_dQ
		# dL_dW_K = X.T @ dL_dK
		# dL_dW_V = X.T @ dL_dV

		# 2. Считаем градиент по общей матрице W_QKV
		X_flat = X.reshape(-1, D)
		dL_dQKV_flat = dL_dQKV_3d.reshape(-1, 3 * D)
		dL_dW_QKV = X_flat.T @ dL_dQKV_flat  # (D, B*L) @ (B*L, 3*D) -> (D, 3*D)

		# 3. Считаем dL_dX (используем всю W_QKV целиком)
		# (B, L, 3*D) @ (3*D, D) -> (B, L, D)
		dL_dX = dL_dQKV_3d @ W_QKV.T

		self.grads = [dL_dW_QKV, dL_dW_O]
		return dL_dX


def scaled_dot_product_attention(Q, K, V, mask=None):
	"""
	Q, K, V как минимум матрицы
	"""
	d = Q.shape[-1]
	logits = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d)
	if mask is not None:
		logits += (mask * -1e9)

	SM = softmax(logits)
	Y = SM @ V
	return Y, SM