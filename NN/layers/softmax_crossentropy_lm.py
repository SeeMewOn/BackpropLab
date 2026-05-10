from NN.functions import softmax
from utils.backend import np

from NN.layer import Layer


class SoftmaxCrossEntropy(Layer):
	"""
	Расчёт функции потерь и нулевого градиента без тензора меток T.

	Вместо того чтобы хранить тензор one-hot меток T, размера
	(B, L, V), где 99.99% данных – это нули, мы храним матрицу меток (B, L)
	"""

	def __init__(self):
		super().__init__()
		self.Y: np.ndarray = np.array([])

	def predict(self, X) -> np.ndarray:
		return softmax(X)  # (B, L, V)

	def forward(self, X: np.ndarray) -> np.ndarray:
		Y = self.predict(X)
		self.Y = Y
		return Y

	def loss(self, T) -> float: # TODO вынести расчёт loss из этого класса
		"""
		Вычисляет скалярный Loss
		T: (B, L) - индексы правильных токенов
		"""
		B, L, V = self.Y.shape
		# Используем продвинутую индексацию numpy/cupy, чтобы достать
		# только нужные вероятности.
		batch_idx = np.arange(B)[:, None]  # (B, 1)
		seq_idx = np.arange(L)[None, :]  # (1, L)

		# Выбираем вероятности правильных классов
		probs = self.Y[batch_idx, seq_idx, T]

		# Добавляем маленькое число (eps), чтобы не было log(0)
		loss = -np.mean(np.log(probs + 1e-10))
		return float(loss)

	def backward(self, T, padding_mask=None) -> np.ndarray:
		"""
	    Вычисляет dL/dX.
	    Формула: (Y_i - 1) если i == правильный класс, иначе Y_i
	    """
		# T (B, L)
		B, L, V = self.Y.shape

		# Создаем сетку индексов
		batch_idx = np.arange(B)[:, None] # (B, 1)
		seq_idx = np.arange(L)[None, :] # (1, L)

		# Вычитаем единицу только там, где стоит правильный токен
		dL_dX = self.Y.copy()  # Копируем вероятности (B, L, V)
		dL_dX[batch_idx, seq_idx, T] -= 1.0

		if padding_mask is not None:
			# Обнуляем градиенты для PAD-токенов по всему dL_dX
			# mask[:, :, None] превращает (B, L) в (B, L, 1) для broadcasting
			dL_dX *= padding_mask[:, :, np.newaxis]
			return dL_dX / np.sum(padding_mask)  # Нормализация на число реальных токенов

		# Нормализуем градиент на размер батча и длину последовательности
		return dL_dX / (B * L)  # (B, L, V)
