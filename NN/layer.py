from utils.backend import np


class Layer:
	def __init__(self, is_training: bool = True):
		self.is_training = is_training       # Режим инференс/обучение
		self.params: list[np.ndarray] = []   # Параметры
		self.grads: list[np.ndarray] = []    # Градиенты

		# Состояние слоя - это накопленные в результате
		# обучения оценки, необходимые на инференсе
		self.state: list[np.ndarray] = []

	def predict(self, X: np.ndarray) -> np.ndarray:
		""" Инференс без "ненужных" вычислений """
		raise NotImplementedError

	def forward(self, X: np.ndarray) -> np.ndarray:
		raise NotImplementedError

	def backward(self, dL_dout: np.ndarray) -> np.ndarray:
		raise NotImplementedError


