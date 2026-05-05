import time

from NN.functions import softmax
from utils.backend import np

from NN.layer import Layer


class MultiHeadSelfAttention(Layer):
	def __init__(self, d_model, n_head):
		super().__init__()
		# Trainable params
		# Xavier init
		limit = np.sqrt(6 / (d_model + d_model))
		W_Q = np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
		# W_Q = np.zeros((d_model, d_model)).astype(np.float32)
		W_K = np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
		W_V = np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
		W_O = np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
		self.params = [W_Q, W_K, W_V, W_O]

		# Hyperparams
		# self.d_model = d_model
		self.n_head = n_head
		self.d_head = int(d_model / n_head)

		# Cache
		self.cache = []

	def predict(self, X: np.ndarray) -> np.ndarray:
		pass

	def forward(self, X, mask=None):
		"""
		X: (batch_size, sequence_length, d_model) или (B, L, D)
		"""
		B, L, D = X.shape
		W_Q, W_K, W_V, W_O = self.params

		# (B, L, D) @ (D, D) -> (B, L, D)
		Q = X @ W_Q
		K = X @ W_K
		V = X @ W_V

		# Подготовка тензоров для входа в scaled_dot_product_attention
		# (B, L, D) -> (B, L, H, d_h)
		Q = Q.reshape(B, L, self.n_head, self.d_head)
		K = K.reshape(B, L, self.n_head, self.d_head)
		V = V.reshape(B, L, self.n_head, self.d_head)

		# (B, L, H, d_h) -> (B, H, L, d_h)
		Q = Q.transpose(0, 2, 1, 3)
		K = K.transpose(0, 2, 1, 3)
		V = V.transpose(0, 2, 1, 3)

		# Подаём полученные тензоры в SDPA. (B, H, L, d_h) -> (B, H, L, d_h)
		SDPA_4d, SM = scaled_dot_product_attention(Q, K, V, mask)

		# (B, H, L, d_h) -> (B, L, H, d_h)
		SDPA_4d = SDPA_4d.transpose(0, 2, 1, 3)

		# (B, L, H, d_h) -> (B, L, D)
		SDPA_3d = SDPA_4d.reshape(B, L, D)

		# (B, L, D) ->  (B, L, D)
		Y = SDPA_3d @ W_O

		# Добавляем данные в кэш для расчёта бэкпропа
		self.cache = [X, Q, K, V, SM, SDPA_3d]
		return Y

	def backward(self, dL_dY):
		W_Q, W_K, W_V, W_O = self.params
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

		# 6. dL_dX
		# (B, L, D) @ (D, D) + (B, L, D) @ (D, D) + (B, L, D) @ (D, D) = (B, L, D)
		dL_dX = dL_dQ @ W_Q.T + dL_dK @ W_K.T + dL_dV @ W_V.T

		# 7. dL_dW_Q, dL_dW_K, dL_dW_V
		# (B, L, D) -> (B*L, D)
		X = X.reshape(-1, D)
		dL_dQ = dL_dQ.reshape(-1, D)
		dL_dK = dL_dK.reshape(-1, D)
		dL_dV = dL_dV.reshape(-1, D)
		# (D, B*L) @ (B*L, D) -> (D, D)
		dL_dW_Q = X.T @ dL_dQ
		dL_dW_K = X.T @ dL_dK
		dL_dW_V = X.T @ dL_dV

		self.grads = [dL_dW_Q, dL_dW_K, dL_dW_V, dL_dW_O]
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


# def self_attention(X, mask=None):
# 	return scaled_dot_product_attention(X, X, X, mask)


if __name__ == '__main__':

	# (B, L, D)
	tensor = np.random.rand(32, 1024, 512)
	mha = MultiHeadSelfAttention(512, 4)

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
