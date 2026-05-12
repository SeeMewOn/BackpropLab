from utils.backend import np

from NN.model import Model


# TODO Удалить устаревший оптимизатор
class OptimizerOld:
	def __init__(
			self,
			model: Model,
			lr: float,  # learning rate
	):
		self.layers_reversed = list(reversed(model.layers))
		self.lr = lr

	def backward(self, T: np.ndarray):
		""" Вычисление градиентов по обучаемым параметрам и сохранение оных """
		out = T  # (56)
		for layer in self.layers_reversed:
			out = layer.backward(out)

	def step(self):
		""" Params update """
		for layer in self.layers_reversed:
			#
			if layer.params:
				layer.params[0] -= self.lr * layer.grads[0]
				layer.params[1] -= self.lr * layer.grads[1]


class Optimizer:
	def __init__(self, params, lr=0.01):
		"""
		:param params: tuple of params and gradients.
		:param lr: learning rate.
		"""
		self.params = params
		self.lr = lr
		self.t = 0  # счётчик шагов

	def step(self):
		raise NotImplementedError


class Adam(Optimizer):
	def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
		super().__init__(params, lr)
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps
		self.wd = weight_decay

		# Инициализируем инерцию и накопление квадратов частных производных для каждого параметра
		self.m = [np.zeros_like(p) for p in self.params[0]]
		self.v = [np.zeros_like(p) for p in self.params[0]]

	def step(self):
		params, grads = self.params
		self.t += 1
		for i, (p, g) in enumerate(zip(params, grads)):
			if np.isnan(g).any():
				raise ValueError(f"NaN detected in gradients at step #{self.t} (param {i})")

			# Weight decay
			if self.wd > 0:
				g_current = g + self.wd * p
			else:
				g_current = g

			# Обновление моментов
			self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_current
			self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g_current ** 2)

			# Корректировка смещения
			m_hat = self.m[i] / (1 - self.beta1 ** self.t)
			v_hat = self.v[i] / (1 - self.beta2 ** self.t)

			# Обновление обучаемых параметров
			p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
