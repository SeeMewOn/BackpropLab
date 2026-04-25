from collections.abc import Callable

import numpy as np

from NN.functions import phi_for_diploma


class NormalEquation:
	def __init__(
			self,
			phi: Callable[[np.ndarray], np.ndarray] = None,
			w: np.ndarray = np.array([]),
	):
		# Training parameters
		self.w = w
		self.phi = phi

	# TODO TEST
	def fit_ridge(self, x_dataset, t_dataset, l):
		"""
		Регуляризация с помощью Ridge Regression.
		:param x_dataset:
		:param t_dataset:
		:param l: Коэффициент регуляризации.
		:return:
		"""
		Phi = np.array([self.phi(x) for x in x_dataset])
		I = np.eye(Phi.shape[1])
		I[0, 0] = 0
		self.w = np.linalg.inv(Phi.T @ Phi + l * I) @ Phi.T @ t_dataset

	def fit(self, x_dataset, t_dataset):
		""" Нет регуляризации"""
		Phi = np.array([self.phi(x) for x in x_dataset])
		self.w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ t_dataset

	def predict(self, x):
		return self.w @ self.phi(x)

	def fit3(self, x_dataset, t_dataset):
		""" Нет регуляризации"""
		Phi = np.array([phi_for_diploma(x) for x in x_dataset])
		self.w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ t_dataset

	def fit4(self, x_dataset, t_dataset, l):
		Phi = np.array([phi_for_diploma(x) for x in x_dataset])
		I = np.eye(len(self.w))
		I[0, 0] = 0
		self.w = np.linalg.inv(Phi.T @ Phi + l * I) @ Phi.T @ t_dataset

	def predict3(self, x):
		return self.w @ phi_for_diploma(x)
