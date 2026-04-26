from matplotlib import pyplot as plt

from NN.functions import im2tensor
from NN.model import Model
from models.CatsVsDogs.train import CnnCatDogV4BN

from utils.backend import np


def visualise_layers(m: Model):
	x_target: np.ndarray = np.zeros((16, 112, 112)).astype(np.float32)
	t_target: np.ndarray = np.array([1, 0])
	X_target = np.array([x_target])
	T_target = np.array([t_target])
	# print("DDD")
	# print(X_target.shape)
	# print(T_target.shape)
	lr = 3
	epochs = 100
	layers = m.layers[4:]
	for ep in range(epochs):
		print(f"iter: {ep}")
		# Forward
		out = X_target
		for layer in layers:
			out = layer.forward(out)
		print(round(float(np.linalg.norm(out - T_target)), 3))

		# Backward
		out = T_target
		for layer in reversed(layers):
			out = layer.backward(out)
		X_target -= lr * out

	img = X_target[0][10].get()
	plt.imshow(
		img,
		cmap='gray'
	)
	plt.show()


# print(model.predict(X_target))
# img = X_target[0].transpose(1, 2, 0).get()
# img = np.clip((img + 0.5), 0, 1)
# plt.imshow(img)
# plt.show()


if __name__ == '__main__':
	model1 = CnnCatDogV4BN()
	model1.load_params("CnnCatDogV4BN/EP_9_PARAMS.npz")

	visualise_layers(model1)

	model2 = CnnCatDogV4BN()
	model2.load_params("CnnCatDogV4BN/EP_15_PARAMS.npz")
