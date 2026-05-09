from utils.backend import np

from NN.layer import Layer


class Dropout(Layer):
	def __init__(self, p=0.5):
		"""
		Слой, для выключения нейронов дальнейших слоёв
		:param p: Вероятность того, что нейрон выключится.
		"""
		super().__init__()
		self.mask: np.ndarray = np.array([])
		self.p = p

	def predict(self, X: np.ndarray) -> np.ndarray:
		return X

	def forward(self, X: np.ndarray) -> np.ndarray:
		if self.is_training:
			# Вероятность сохранить нейрон = (1 - p)
			keep_prob = 1.0 - self.p

			self.mask = (np.random.rand(*X.shape) > self.p) / keep_prob
			return X * self.mask
		return X

	def backward(self, dL_dY: np.ndarray, calc_grads: bool = True) -> np.ndarray:
		if self.is_training:
			return (dL_dY * self.mask) / (1.0 - self.p)
		return dL_dY
