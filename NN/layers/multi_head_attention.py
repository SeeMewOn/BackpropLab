from NN.functions import softmax
from utils.backend import np

from NN.layer import Layer


class MultiHeadAttention(Layer):
	pass


def scaled_dot_product_attention(Q, K, V, mask=None):
	"""
	Q, K, V как минимум матрицы
	"""
	d = Q.shape[-1]
	logits = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d)
	if mask is not None:
		logits += (mask * -1e9)

	Y = softmax(logits) @ V
	return Y

def self_attention(X, mask=None):
	return scaled_dot_product_attention(X, X, X, mask)



if __name__ == '__main__':
	vector = np.array([1, 2])
	matrix = np.array([[1, 2],
	                   [3, 4]])
	tensor3d = np.array([
		[[5, 6],
		 [70, 8]],
		[[9, 10],
		 [11, 12]]
	])

	q = np.array([
		[[5, 6],
		 [70, 8]],
		[[9, 8],
		 [11, 15]]
	])
	k = np.array([
		[[0, 6],
		 [2, 8]],
		[[0, 0],
		 [11, 12]]
	])
	v = np.array([
		[[5, 0],
		 [70, 8]],
		[[0, 10],
		 [11, 0]]
	])
	print(self_attention(tensor3d))
	print()
	print(scaled_dot_product_attention(tensor3d, tensor3d, tensor3d))
	print()
	print(scaled_dot_product_attention(q, k, v))
