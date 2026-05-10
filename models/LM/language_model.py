import time

from NN.functions import get_combined_mask
from NN.layers.dense import Dense
from NN.layers.input_embedding import InputEmbedding
from NN.layers.layer_norm import LayerNorm
from NN.layers.positional_embedding import PositionalEmbedding
from NN.layers.softmax_crossentropy_lm import SoftmaxCrossEntropy
from NN.layers.transformer_block import TransformerBlock
from utils.backend import np


class LanguageModel:
	def __init__(
			self,
			vocab_size=16000,
			d_model=512,
			context_size=1024,
			dtype=np.float32,
			pad_token_id=0
	):
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
		self.softmax_cross_entropy = SoftmaxCrossEntropy()

	def forward(self, X, mask=None):
		# Маска для scaled dot-product attention
		# mask = get_combined_mask(X, pad_token_id=0)

		out = self.input_embedding.forward(X)
		out = self.positional_embedding.forward(out)
		out = self.transformer_block_1.forward(out, mask)
		out = self.transformer_block_2.forward(out, mask)
		out = self.transformer_block_3.forward(out, mask)
		out = self.transformer_block_4.forward(out, mask)
		out = self.transformer_block_5.forward(out, mask)
		out = self.transformer_block_6.forward(out, mask)
		out = self.layer_norm.forward(out)
		out = self.dense.forward(out)
		out = self.softmax_cross_entropy.forward(out)
		return out

	def backward(self, T, padding_mask=None):
		out = self.softmax_cross_entropy.backward(T, padding_mask)
		out = self.dense.backward(out)
		out = self.layer_norm.backward(out)
		out = self.transformer_block_6.backward(out)
		out = self.transformer_block_5.backward(out)
		out = self.transformer_block_4.backward(out)
		out = self.transformer_block_3.backward(out)
		out = self.transformer_block_2.backward(out)
		out = self.transformer_block_1.backward(out)
		out = self.positional_embedding.backward(out)
		out = self.input_embedding.backward(out)

	def eval(self):
		self.input_embedding.eval()
		self.positional_embedding.eval()
		self.transformer_block_1.eval()
		self.transformer_block_2.eval()
		self.transformer_block_3.eval()
		self.transformer_block_4.eval()
		self.transformer_block_5.eval()
		self.transformer_block_6.eval()
		self.layer_norm.eval()
		self.dense.eval()
		self.softmax_cross_entropy.eval()

	def tarin(self):
		self.input_embedding.train()
		self.positional_embedding.train()
		self.transformer_block_1.train()
		self.transformer_block_2.train()
		self.transformer_block_3.train()
		self.transformer_block_4.train()
		self.transformer_block_5.train()
		self.transformer_block_6.train()
		self.layer_norm.train()
		self.dense.train()
		self.softmax_cross_entropy.train()

	def zero_grad(self):
		self.input_embedding.zero_grad()
		self.positional_embedding.zero_grad()
		self.transformer_block_1.zero_grad()
		self.transformer_block_2.zero_grad()
		self.transformer_block_3.zero_grad()
		self.transformer_block_4.zero_grad()
		self.transformer_block_5.zero_grad()
		self.transformer_block_6.zero_grad()
		self.layer_norm.zero_grad()
		self.dense.zero_grad()
		self.softmax_cross_entropy.zero_grad()

if __name__ == '__main__':
	lm = LanguageModel()

	start = time.time()
	for t in range(30):
		x = np.random.randint(low=0, high=15999, size=(8, 1024))  # (B, L)
		lm.forward(x)
		print(f"\r{t}", end="")

	print()
	end = time.time()
	print(f"Forward time: {end - start}")

	# Backward test
	start = time.time()
	for t in range(30):
		grad = np.random.rand(8, 1024, 16000).astype(np.float32)
		lm.backward(grad)
		print(f"\r{t}", end="")
	# print(t)

	print()
	end = time.time()
	print(f"Backward time: {end - start}")
