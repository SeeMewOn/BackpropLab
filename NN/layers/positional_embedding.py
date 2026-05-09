import time

from NN.layer import Layer
from utils.backend import np


class PositionalEmbedding(Layer):
	"""Non-trainable sin-cos PositionalEmbedding"""
	def __init__(self, context_size: int, d_model: int, C=10000):
		super().__init__()

		# Инициализация матрицы позиционных эмбеддингов
		self.pe = np.zeros((context_size, d_model)).astype(np.float32)  # (L, D)
		pos = np.arange(0, context_size).reshape(-1, 1).astype(np.float32)
		div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * -(np.log(C) / d_model))
		self.pe[:, 0::2] = np.sin(pos * div_term)
		self.pe[:, 1::2] = np.cos(pos * div_term)

	def predict(self, X: np.ndarray) -> np.ndarray:
		# X (B, L, D)
		return X + self.pe

	def forward(self, X: np.ndarray) -> np.ndarray:
		return self.predict(X)

	def backward(self, dL_dY: np.ndarray) -> np.ndarray:
		return dL_dY

if __name__ == '__main__':
	print(np.arange(0, 10).reshape(-1, 1).astype(np.float32))

