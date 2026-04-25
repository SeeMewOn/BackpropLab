from NN.layer import Layer
from utils.backend import np


class Model:
	def __init__(self):
		self.layers: list[Layer] = []

	def forward(self, X: np.ndarray) -> np.ndarray:
		out = X
		for layer in self.layers:
			out = layer.forward(out)

		return out

	def predict(self, X: np.ndarray):
		out = X
		for layer in self.layers:
			out = layer.predict(out)

		return out

	def save_params(self, path: str):
		params_dict = {}
		for i, layer in enumerate(self.layers):
			# Write params
			if layer.params:
				params_dict[f"layer_{i}_W"] = layer.params[0]
				params_dict[f"layer_{i}_b"] = layer.params[1]

			# Write states
			if layer.state:
				for j, s in enumerate(layer.state):
					params_dict[f"layer_{i}_s_{j}"] = s

		np.savez_compressed(path, **params_dict)

	def load_params(self, path: str):
		data = np.load(path)
		for i, layer in enumerate(self.layers):
			# Read params
			if layer.params:
				W = data[f"layer_{i}_W"]
				b = data[f"layer_{i}_b"]
				layer.params = [W, b]

			# Read states
			if layer.state:
				state = []
				for j, s in enumerate(layer.state):
					state.append(data[f"layer_{i}_s_{j}"])
				layer.state = state
