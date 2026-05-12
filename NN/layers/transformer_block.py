import time

from NN.layers.gelu import GELU
from utils.backend import np
from NN.layer import Layer
from NN.layers.dense import Dense
from NN.layers.dropout import Dropout
from NN.layers.layer_norm import LayerNorm
from NN.layers.multi_head_attention import MultiHeadSelfAttention


class TransformerBlock(Layer):
	def __init__(self, d_model, n_head):
		super().__init__()
		# Attention module & skip connection
		self.ln1 = LayerNorm(d_model)
		self.mha = MultiHeadSelfAttention(d_model, n_head)
		self.dropout1 = Dropout(0.1)

		# Linear module & skip connection
		self.ln2 = LayerNorm(d_model)
		self.dense1 = Dense(d_model, 4 * d_model)
		self.gelu = GELU()
		self.dense2 = Dense(4 * d_model, d_model)
		self.dropout2 = Dropout(0.1)

		# All layers
		self.layers = [
			self.ln1, self.mha, self.dropout1,
			self.ln2, self.dense1, self.gelu, self.dense2, self.dropout2
		]

	def predict(self, X, mask=None):
		# Attention module & skip connection
		out1 = self.ln1.predict(X)
		out1 = self.mha.predict(out1, mask)  # Маска
		out1 = self.dropout1.predict(out1)
		out1 += X

		# Linear module & skip connection
		out2 = self.ln2.predict(out1)
		out2 = self.dense1.predict(out2)
		out2 = self.gelu.predict(out2)
		out2 = self.dense2.predict(out2)
		out2 = self.dropout2.predict(out2)
		out2 += out1
		return out2

	def forward(self, X, mask=None):
		# Attention module & skip connection
		out1 = self.ln1.forward(X)
		out1 = self.mha.forward(out1, mask)  # Маска
		out1 = self.dropout1.forward(out1)
		out1 += X

		# Linear module & skip connection
		out2 = self.ln2.forward(out1)
		out2 = self.dense1.forward(out2)
		out2 = self.gelu.forward(out2)
		out2 = self.dense2.forward(out2)
		out2 = self.dropout2.forward(out2)
		out2 += out1
		return out2

	def backward(self, dL_dY):
		# Linear module & skip connection
		grad1 = self.dropout2.backward(dL_dY)
		grad1 = self.dense2.backward(grad1)
		grad1 = self.gelu.backward(grad1)
		grad1 = self.dense1.backward(grad1)
		grad1 = self.ln2.backward(grad1)
		grad1 += dL_dY

		# Attention module & skip connection
		grad2 = self.dropout1.backward(grad1)
		grad2 = self.mha.backward(grad2)
		grad2 = self.ln1.backward(grad2)
		grad2 += grad1
		return grad2

	def train(self):
		super().train()
		for layer in self.layers:
			layer.train()

	def eval(self):
		super().eval()
		for layer in self.layers:
			layer.eval()

	def zero_grad(self):
		for layer in self.layers:
			layer.zero_grad()

	def get_params(self):
		params = []
		grads = []
		for layer in self.layers:
			p, g = layer.get_params()
			params.extend(p)
			grads.extend(g)
		return params, grads


if __name__ == '__main__':

	# (B, L, D)
	B, L, D, H = 32, 1024, 512, 8

	transformer_block = TransformerBlock(D, H)

	# Forward test
	start = time.time()
	for t in range(10):
		tensor = np.random.rand(B, L, D)
		transformer_block.forward(tensor)
		print(f"\r{t}", end="")

	end = time.time()
	print()
	print(f"Forward time: {end - start}")

	# Backward test
	start = time.time()
	for t in range(10):
		tensor = np.random.rand(B, L, D)
		transformer_block.backward(tensor)
		print(f"\r{t}", end="")

	end = time.time()
	print()
	print(f"Backward time: {end - start}")
