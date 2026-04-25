from utils.backend import np

from NN.functions import shuffle_dataset, cross_entropy


class ActivationFunction:
	@staticmethod
	def fn(A):
		raise NotImplementedError

	@staticmethod
	def deriv(A):
		raise NotImplementedError


class RELU(ActivationFunction):
	@staticmethod
	def fn(A: np.ndarray):
		return np.maximum(0, A)

	@staticmethod
	def deriv(A: np.ndarray):
		return (A > 0).astype(int)

class LinearActivation(ActivationFunction):
	@staticmethod
	def fn(A: np.ndarray):
		return A

	@staticmethod
	def deriv(A: np.ndarray):
		return np.ones_like(A)

class Softmax(ActivationFunction):
	@staticmethod
	def fn(A: np.ndarray):
		# print(A.shape)
		A_exp = np.exp(A - np.max(A, axis=1, keepdims=True))
		S = np.sum(A_exp, axis=1, keepdims=True)
		return A_exp / S

	@staticmethod
	def deriv(A: np.ndarray):
		return 0


class Layer:
	def __init__(
			self,
			input_size: int,
			output_size: int,
			activation_fn: ActivationFunction
			# activation_fn: Callable[[np.ndarray], np.ndarray] = lambda A: A
	):
		""" l-й слой MLP """
		self.activation_fn = activation_fn
		self.W: np.ndarray = np.random.randn(output_size, input_size).astype(np.float32) * 0.01  # (M_l, M_(l-1))
		self.b: np.ndarray = np.zeros(output_size, dtype=np.float32)  # (M_l,)
		self.grad_W = np.array([])  # (M_l, M_(l-1))
		self.grad_b = np.array([])  # (M_l,)
		self.X = np.array([])  # (N, M_(l-1))
		self.deriv_f_A = np.array([])  # (N, M_l)

	def output(self, X):
		A = X @ self.W.T + self.b  # (N, M_l)
		Z = self.activation_fn.fn(A)  # (N, M_l)
		return Z

	def forward(self, X):
		# Считаем выход слоя по батчу
		A = X @ self.W.T + self.b  # (N, M_l)
		Z = self.activation_fn.fn(A)  # (N, M_l)
		# Сохраняем вход для этого слоя (по батчу)
		self.X = X
		# Сохраняем матрицу производных функции активации
		self.deriv_f_A = self.activation_fn.deriv(A)
		return Z

	def backward(self, grad_a: np.ndarray):
		# grad_a: (N, M_l)
		N = grad_a.shape[0]
		grad_b = grad_a.mean(axis=0)  # (M_l,)
		grad_W = grad_a.T @ self.X / N  # (M_l, M_{l-1})
		self.grad_W = grad_W
		self.grad_b = grad_b
		# grad_Z_previous = np.dot(self.W.T, s)
		grad_Z_previous = grad_a @ self.W
		return grad_Z_previous


class MLP:
	def __init__(
			self,
			# Hyperparameters
			learning_rate: np.float32 = 0,
			epoch_count: int = 0,
			training_ratio: float = 0,
			validation_ratio: float = 0,
			batch_size: int = 0,

			# Data parameters
			means: np.ndarray = np.array([]),
			stds: np.ndarray = np.array([]),

			# Optimize
			log_metric_step: int = 100
	):
		# Training params
		self.layers: list[Layer] = []

		# Hyperparameters
		self.epoch_count = epoch_count
		self.learning_rate = learning_rate
		self.training_ratio = training_ratio
		self.validation_ratio = validation_ratio
		self.batch_size = batch_size

		# Data parameters
		self.means = means
		self.stds = stds

		# Statistics
		self.training_errors = []
		self.validation_errors = []
		self.accuracies = []
		self.confusion_matrix = None

		# Optimize
		self.log_metric_step = log_metric_step

	def add(self, layer: Layer):
		self.layers.append(layer)

	def forward(self, X):
		for layer in self.layers:
			X = layer.forward(X)
		return X

	def backward(self, grad_a):
		layers_reversed = list(reversed(self.layers))
		for i, layer in enumerate(layers_reversed):
			# Вычисление градиента по Z и по a предыдущего слоя
			grad_Z_previous = layer.backward(grad_a)

			if i != len(layers_reversed) - 1:
				grad_a = np.multiply(layers_reversed[i + 1].deriv_f_A, grad_Z_previous) # TODO float32


	def update(self):
		for layer in self.layers:
			layer.W -= self.learning_rate * layer.grad_W
			layer.b -= self.learning_rate * layer.grad_b


	def predict(self, X):
		for layer in self.layers:
			X = layer.output(X)
		return X

	def fit(self, X: np.ndarray, T: np.ndarray):
		# Подготовка данных
		training_set_size = int(self.training_ratio * len(X))
		validation_set_size = int(self.validation_ratio * len(X))
		X, T = shuffle_dataset(X, T)
		X_training = X[:training_set_size]
		T_training = T[:training_set_size]
		X_validation = X[training_set_size:training_set_size + validation_set_size]
		T_validation = T[training_set_size:training_set_size + validation_set_size]
		X_test = X[training_set_size + validation_set_size:]
		T_test = T[training_set_size + validation_set_size:]

		# Нормализация данных
		# self.means = X_training.mean(axis=0)  # (M,)
		# self.stds = X_training.std(axis=0)  # (M,)
		# TODO
		self.means = 0  # (M,)
		self.stds = 255  # (M,)
		X_training = (X_training - self.means) / self.stds
		X_validation = (X_validation - self.means) / self.stds

		# Количество корректировок весов в одной эпохе
		batch_steps = X_training.shape[0] // self.batch_size

		for epoch in range(self.epoch_count):
			# Перемешиваем тренировочный датасет
			X_training, T_training = shuffle_dataset(X_training, T_training)

			# Пробегаемся по батчам
			for i in range(batch_steps):
				X_batch = X_training[self.batch_size * i: self.batch_size * (i + 1)]
				T_batch = T_training[self.batch_size * i: self.batch_size * (i + 1)]

				# 1. Forward
				Y_batch = self.forward(X_batch)

				# 2. Backward
				self.backward(Y_batch - T_batch)

				# 3. Training params update
				self.update()

			# Вычисление ошибки, если нужно
			if epoch % self.log_metric_step == 0:
				Y_validation = self.predict(X_validation)
				Y_training = self.predict(X_training)
				confusion_matrix = confusion_matrix(
					Y_validation,
					T_validation,
				)
				accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

				self.validation_errors.append(float(cross_entropy(Y_validation, T_validation)))
				self.training_errors.append(float(cross_entropy(Y_training, T_training)))
				self.accuracies.append(float(accuracy))
				self.confusion_matrix = confusion_matrix
				print(
					f"\rЭпоха {epoch} | "
					f"Training Error: {self.training_errors[-1]:.2e} | "
					f"Validation Error: {self.validation_errors[-1]:.2e} | "
					f"Accuracy: {self.accuracies[-1] * 100:.2e}",
					end=""
				)

