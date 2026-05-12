from utils.backend import np


class Layer:
	def __init__(self, is_training: bool = True):
		self.is_training = is_training  # Режим инференс/обучение
		self.params: list[np.ndarray] = []  # Параметры
		self.grads: list[np.ndarray] = []  # Градиенты

		# Состояние слоя - это накопленные в результате
		# обучения оценки, необходимые на инференсе
		self.state: list[np.ndarray] = []

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""
		Инференс без "ненужных" вычислений.
		Используется ТОЛЬКО при is_training == False
		"""
		raise NotImplementedError

	def forward(self, X: np.ndarray) -> np.ndarray:
		""" Вычисление выхода """
		raise NotImplementedError

	def backward(self, dL_dY: np.ndarray) -> np.ndarray:
		raise NotImplementedError

	def train(self):
		""" Переключение модели в режим обучения. """
		self.is_training = True

	def eval(self):
		""" Переключение модели в режим инференса. """
		self.is_training = False

	def zero_grad(self):
		""" Обнуление всех градиентов. """
		for grad in self.grads:
			grad.fill(0.0)

	def get_params(self):
		"""
		Возвращает параметры и градиенты целевой
		функции по ним для данного слоя. Это нужно для
		последующей их передачи оптимизатору, чтобы
		осуществить шаг обновления параметров.
		"""
		return self.params, self.grads
