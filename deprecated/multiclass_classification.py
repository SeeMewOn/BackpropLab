import numpy as np

from NN.functions import grad_log_reg, shuffle_dataset, error_log_reg, softmax_vector, softmax


class MulticlassClassification:
	def __init__(  # TODO Batch
			self,
			# Training parameters
			W: np.ndarray,
			b: np.ndarray,

			# Hyperparameters
			gamma: np.float32 = 0,
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
		"""
		Классификация
		:param W: Матраца весов (K×M)
		:param b: Вектор Bias (K)
		:param gamma: Скорость
		"""
		# Training parameters
		self.W = W
		self.b = b
		self.gamma = gamma

		# Hyperparameters
		self.epoch_count = epoch_count
		self.training_ratio = training_ratio
		self.validation_ratio = validation_ratio
		self.batch_size = batch_size

		# Data parameters
		self.means = means
		self.stds = stds

		# Statistics
		self.training_errors = []
		self.validation_errors = []
		self.confusion_matrix = None
		self.accuracies = []

		# Optimize
		self.log_metric_step = log_metric_step

	def log_metrics(self, X_training, T_training, X_validation, T_validation, epoch):
		# Расчёт метрик
		Y_validation = softmax((X_validation - self.means) / self.stds @ self.W.T + self.b)
		confusion_matrix = confusion_matrix(
			Y_validation,
			T_validation,
		)
		accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
		training_error = error_log_reg(self.W, self.b, X_training, T_training)
		validation_error = error_log_reg(self.W, self.b, (X_validation - self.means) / self.stds, T_validation)

		# Добавление в статистику
		self.accuracies.append(accuracy)
		self.training_errors.append(training_error)
		self.validation_errors.append(validation_error)
		self.confusion_matrix = confusion_matrix

		# Отображение метрик во время обучения
		print(
			f"\rЭпоха {epoch} | "
			f"Training Error: {self.training_errors[-1]:.2e} | "
			f"Validation Error: {self.validation_errors[-1]:.2e} | "
			f"Accuracy: {round(self.accuracies[-1] * 100, 2)}%",
			end=""
		)

	def fit(self, X: np.ndarray, T: np.ndarray):
		# todo сделать подготовку данных отдельной функцией, включающую баланс классов
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

		# Нормализация тренировочного датасета
		self.means = X_training.mean(axis=0)  # (M,)
		self.stds = X_training.std(axis=0)  # (M,)
		X_training = (X_training - self.means) / self.stds

		# Количество корректировок весов в одной эпохе
		batch_steps = X_training.shape[0] // self.batch_size

		for epoch in range(self.epoch_count):
			# Перемешиваем тренировочный датасет
			X_training, T_training = shuffle_dataset(X_training, T_training)

			# Пробегаемся по батчам
			for i in range(batch_steps):
				# Вычисляем градиент, точнее матрицу частных производных W и вектор частных производных b
				grad_W, grad_b = grad_log_reg(
					self.W, self.b,
					X_training[self.batch_size * i: self.batch_size * (i + 1)],
					T_training[self.batch_size * i: self.batch_size * (i + 1)],
				)

				# Корректировка обучаемых параметров
				self.W -= self.gamma * grad_W
				self.b -= self.gamma * grad_b

			# Логирование метрик
			if epoch % self.log_metric_step == 1:
				self.log_metrics(X_training, T_training, X_validation, T_validation, epoch)



	def predict(self, x):
		x = (x - self.means) / self.stds
		return softmax_vector(self.W @ x + self.b)
