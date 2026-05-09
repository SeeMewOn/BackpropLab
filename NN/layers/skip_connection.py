from NN.layer import Layer
from utils.backend import np


class SkipConnection(Layer):
	def __init__(self, layers: list[Layer]):
		super().__init__()
		self.layers = layers

	def predict(self, X):
		out = X
		for layer in self.layers:
			out = layer.predict(out)

		return out + X

	def forward(self, X):
		out = X
		for layer in self.layers:
			out = layer.forward(out)

		return out + X

	def backward(self, dL_dY):
		out = dL_dY
		for layer in reversed(self.layers):
			out = layer.backward(out)

		return out + dL_dY