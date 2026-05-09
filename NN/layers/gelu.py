from utils.backend import np

from NN.layer import Layer


class GELU(Layer):
	def __init__(self):
		super().__init__()
		self.X = None

	def predict(self, X: np.ndarray) -> np.ndarray:
		return 0.5 * X * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (X + 0.044715 * X ** 3)))

	def forward(self, X: np.ndarray) -> np.ndarray:
		if self.is_training:
			self.X = X
		return self.predict(X)

	def backward(self, dL_dY: np.ndarray) -> np.ndarray:
		X = self.X
		# Чтобы не считать корень и константы сто раз
		sqrt_2_pi = np.sqrt(2.0 / np.pi)

		# Внутренняя часть тангенса
		inner = sqrt_2_pi * (X + 0.044715 * X ** 3)
		tanh_inner = np.tanh(inner)

		# Производная GELU(x) через tanh аппроксимацию:
		# d/dx GELU(x) = 0.5 * (1 + tanh(inner)) +
		#                0.5 * x * (1 - tanh^2(inner)) * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)

		term1 = 0.5 * (1.0 + tanh_inner)
		term2 = 0.5 * X * (1.0 - tanh_inner ** 2) * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * X ** 2)

		return dL_dY * (term1 + term2)