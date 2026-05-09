import time

from NN.layer import Layer
from utils.backend import np


class InputEmbedding(Layer):
	def __init__(self, vocab_size: int, d_model: int, shared_params:dict=None):
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
		# Trainable params
		if shared_params:
			self.params = shared_params["params"]
			self.grads = shared_params["grads"]
		else:
			# Random init
			W = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.01
			self.params = [W]
			self.grads = [np.zeros_like(W)]

		# Cache
		self.X = None

	def _lookup(self, X: np.ndarray) -> np.ndarray:
		# X имеет размер (B, L), где B - batch, L - context size
		# Преобразование X (B, L) в тензор эмбеддингов Y (B, L, D)
		return self.params[0][X]  # (B, L, D)

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self._lookup(X)

	def forward(self, X: np.ndarray) -> np.ndarray:
		self.X = X  # Сохраняем индексы для градиента
		return self._lookup(X)

	def backward(self, dL_dY: np.ndarray):
		"""
		dL_dY: градиент от следующего слоя (B, L, D)
		"""
		dL_dW = np.zeros_like(self.params[0])  # (V, D) Матрица нулей

		# Векторизованное накопление градиентов
		# Для каждой позиции в батче, где стоял индекс i,
		# мы прибавляем градиент dL_dY к i-й строке dL_dW.

		# Почему не "dL_dW[self.X] = dL_dY"?
		#
		# Если в X (B, L) были повторяющиеся id токенов,
		# например в матрице X было 3 повторяющихся числа,
		# то рассчитывая forward мы трижды достали из W (V, D)
		# один и тот же эмбеддинг (пусть он стоит на 10-й позиции,
		# то есть мы трижды достали эмбеддинг W[10,:]) и
		# трижды положили его в выходной тензор Y (B, L, D).
		#
		# В backward приходит 3 разных градиента для этих
		# трех позиций. Если написать dL_dW[self.X] = dL_dY,
		# то мы просто трижды перезапишем 10-ю строку в dL_dW (dL_dW[10,:]).
		# В итоге там останется градиент только от последнего вхождения слова.
		#
		# np.add.at(массив, индексы, что_прибавляем)
		# Она берет индекс из self.X
		# - Берет соответствующий вектор из dL_dY;
		# - Прибавляет его к строке в dL_dW;
		# - Если индекс повторяется, она прибавит следующее значение к уже измененному результату.

		np.add.at(dL_dW, self.X, dL_dY)

		# self.grads = [dL_dW] (OLD)
		self.grads[0] += dL_dW

		# dL_dX не вычисляется, так как X - это вход в
		# нейронную сеть - батч последовательностей token id.
		return None


if __name__ == '__main__':
	x = np.random.randint(low=0, high=15999, size=(32, 1024))  # (B, L)
	grad = np.random.rand(32, 1024, 512).astype(np.float32)  # (B, L, D)
	emb = InputEmbedding(16000, 512)

	# Forward test
	start = time.time()
	for t in range(100):
		emb.forward(x)
		print(f"\r{t}", end="")

	print()
	end = time.time()
	print(f"Forward time: {end - start}")

	# Backward test
	start = time.time()
	for t in range(100):
		emb.backward(grad)
		print(f"\r{t}", end="")

	print()
	end = time.time()
	print(f"Backward time: {end - start}")
