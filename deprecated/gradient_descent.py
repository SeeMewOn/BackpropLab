import numpy as np


from NN.functions import grad_lin_reg, error, phi_polynom, shuffle_dataset
from deprecated.enums import RegMode


class GradientDescent:
	def __init__(
			self,
			w: np.ndarray,
			x_std=None,
			x_mean=None,
			gamma: float=0.01,
			epoch_count: int=1000,
			training_ratio: float = 0.5,
			validation_ratio: float = 0.5,
			reg_mode: RegMode = RegMode.LEAST_SQUARE_REGRESSION,
			l: float | None = None,
	):
		"""
		Модель, использующая градиентный спуск.
		:param w: Вектор весов.
		:param x_std: Дисперсия.
		:param x_mean: Среднее значение икса.
		:param gamma: Скорость обучения.
		:param epoch_count: Количество эпох обучения.
		:param training_ratio: Отношение количества элементов обучающей выборки к размеру датасета.
		:param validation_ratio: Отношение количества элементов валидационной выборки к размеру датасета.
		:param reg_mode: Способ регуляризации.
		:param l: Коэффициент регуляризации
		"""
		# Training parameters
		self.stds = None
		self.means = None
		self.w = w
		self.x_std = x_std
		self.x_mean = x_mean

		# Hyperparameters
		self.gamma = gamma
		self.epoch_count = epoch_count
		self.M = len(self.w)
		self.training_ratio = training_ratio
		self.validation_ratio = validation_ratio
		self.reg_mode = reg_mode
		self.l = l

		# Statistics
		self.training_errors = []
		self.validation_errors = []
		self.test_error = 0
		self.weights = [w.copy()]

	def fit(self, x_dataset: np.ndarray, t_dataset: np.ndarray):
		# Подготовка данных
		# x_dataset, t_dataset = shuffle_dataset(x_dataset, t_dataset)

		training_set_size = int(self.training_ratio * len(x_dataset))
		validation_set_size = int(self.validation_ratio * len(x_dataset))
		x_training_set = x_dataset[:training_set_size]
		t_training_set = t_dataset[:training_set_size]
		x_validation_set = x_dataset[training_set_size:training_set_size + validation_set_size]
		t_validation_set = t_dataset[training_set_size:training_set_size + validation_set_size]
		x_test_set = x_dataset[training_set_size + validation_set_size:]
		t_test_set = t_dataset[training_set_size + validation_set_size:]

		# Матрица плана
		Phi = np.array(
			[
				phi_polynom(x, self.M - 1) for x in x_training_set
			]
		)

		# Нормализация данных
		self.means = Phi.mean(axis=0)
		self.stds = Phi.std(axis=0)
		self.means[0] = 0
		self.stds[0] = 1

		Phi_normalized = (Phi - self.means) / self.stds

		# self.x_mean = np.mean(x_training_set)
		# self.x_std = np.std(x_training_set)

		for epoch in range(self.epoch_count):
			# Перемешиваем тренировочный датасет
			Phi_normalized, t_training_set = shuffle_dataset(Phi_normalized, t_training_set)

			# Вычисление градиента
			g = self._get_gradient(t_training_set, Phi_normalized)

			# Корректировка весов
			self.w -= self.gamma * g

			# Добавление данных
			training_error = self._get_training_error(t_training_set, Phi_normalized)
			validation_error = self._get_validation_error(x_validation_set, t_validation_set)

			self.training_errors.append(training_error)
			self.validation_errors.append(validation_error)
			self.weights.append(self.w.copy())

			# Отображение во время обучения
			print(
				f"\rЭпоха {epoch} | "
				f"Training Error: {training_error:.2e} | "
				f"Validation Error: {validation_error:.2e}",
				end=""
			)

		self.test_error = self._get_test_error(x_test_set, t_test_set)

	def predict(self, x):
		# x_norm = (x - self.x_mean) / self.x_std
		phi_normalized = (phi_polynom(x, self.M - 1) - self.means) / self.stds
		return self.w @ phi_normalized

	def _get_gradient(self, t_training_set, Phi):
		return grad_lin_reg(self.w, t_training_set, Phi, self.reg_mode, self.l)

	def _get_training_error(self, t_training_set, Phi):
		return error(self.w, t_training_set, Phi, self.reg_mode, self.l) / len(t_training_set)

	def _get_validation_error(self, x_validation_set, t_validation_set):
		Phi = np.array([phi_polynom(x, self.M - 1) for x in x_validation_set])
		Phi_normalized = (Phi - self.means) / self.stds
		e = error(self.w, t_validation_set, Phi_normalized, self.reg_mode, self.l) / len(t_validation_set)
		return e

	def _get_test_error(self, x_test_set, t_test_set):
		Phi = np.array([phi_polynom(x, self.M - 1) for x in x_test_set])
		Phi_normalized = (Phi - self.means) / self.stds
		e = error(self.w, t_test_set, Phi_normalized, self.reg_mode, self.l) / len(t_test_set)
		return e
