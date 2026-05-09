import time

from NN.layers.dense import Dense
from NN.layers.input_embedding import InputEmbedding
from NN.layers.layer_norm import LayerNorm
from NN.layers.positional_embedding import PositionalEmbedding
from NN.layers.transformer_block import TransformerBlock
from utils.backend import np


class LanguageModel:
	def __init__(self, vocab_size=16000, d_model=512, context_size=1024):
		# Weight Tying. Общие параметры и градиенты по ним
		# для первого слоя языковой модели (InputEmbedding)
		# и предпоследнего линейного слоя (Dense)
		W = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.01
		b = np.zeros(vocab_size).astype(np.float32)
		dL_dW = np.zeros_like(W)
		dL_db = np.zeros_like(b)

		# Слои модели
		self.input_embedding = InputEmbedding(vocab_size, d_model, shared_params={"params": [W], "grads": [dL_dW]})
		self.positional_embedding = PositionalEmbedding(context_size, d_model)
		self.transformer_block_1 = TransformerBlock(d_model, 8)
		self.transformer_block_2 = TransformerBlock(d_model, 8)
		self.transformer_block_3 = TransformerBlock(d_model, 8)
		self.transformer_block_4 = TransformerBlock(d_model, 8)
		self.transformer_block_5 = TransformerBlock(d_model, 8)
		self.transformer_block_6 = TransformerBlock(d_model, 8)
		self.layer_norm = LayerNorm(d_model)
		self.dense = Dense(d_model, vocab_size, shared_params={"params": [W, b], "grads": [dL_dW, dL_db]})

	def test_forward(self, X):
		out = self.input_embedding.forward(X)
		out = self.positional_embedding.forward(out)
		out = self.transformer_block_1.forward(out)
		out = self.transformer_block_2.forward(out)
		out = self.transformer_block_3.forward(out)
		out = self.transformer_block_4.forward(out)
		out = self.transformer_block_5.forward(out)
		out = self.transformer_block_6.forward(out)
		out = self.layer_norm.forward(out)
		out = self.dense.forward(out)

	def test_backward(self, dL_dY):
		out = self.dense.backward(dL_dY)
		out = self.layer_norm.backward(out)
		out = self.transformer_block_6.backward(out)
		out = self.transformer_block_5.backward(out)
		out = self.transformer_block_4.backward(out)
		out = self.transformer_block_3.backward(out)
		out = self.transformer_block_2.backward(out)
		out = self.transformer_block_1.backward(out)
		out = self.positional_embedding.backward(out)
		out = self.input_embedding.backward(out)











if __name__ == '__main__':
	lm = LanguageModel()


	start = time.time()
	for t in range(30):
		x = np.random.randint(low=0, high=15999, size=(8, 1024))  # (B, L)
		lm.test_forward(x)
		print(f"\r{t}", end="")

	print()
	end = time.time()
	print(f"Forward time: {end - start}")

	# Backward test
	start = time.time()
	for t in range(30):
		grad = np.random.rand(8, 1024, 16000).astype(np.float32)
		lm.test_backward(grad)
		print(f"\r{t}", end="")
		# print(t)

	print()
	end = time.time()
	print(f"Backward time: {end - start}")
