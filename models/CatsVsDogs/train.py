import os
import time

from NN.layers.batch_norm import BatchNorm
from utils.backend import np

from matplotlib import pyplot as plt

from NN.layers.conv import Conv
from NN.layers.dense import Dense
from NN.layers.dropout import Dropout
from NN.layers.flatten import Flatten
from NN.layers.maxpool import MaxPool
from NN.layers.relu import ReLU
from NN.layers.softmax_crossentropy import SoftmaxCrossEntropy
from NN.layers.zeropad import ZeroPad
from NN.model import Model
from NN.optimizer import Optimizer
from NN.functions import shuffle_dataset, confusion_matrix, cross_entropy, im2tensor


# 0 - это кот, 1 - это собак


def load_dataset(start: int, stop: int, target_size=(64, 64)):
	X = []
	T_cats = np.zeros((stop - start, 2))
	T_cats[:, 0:1] = 1
	T_dogs = np.zeros((stop - start, 2))
	T_dogs[:, 1:2] = 1

	for i in range(start, stop):
		X.append(im2tensor(f"../../../data/PetImages/Cat/{i}.jpg", target_size))
		print(f"\rКотик #{i}", end="")

	print()

	for j in range(start, stop):
		X.append(im2tensor(f"../../../data/PetImages/Dog/{j}.jpg", target_size))
		print(f"\rПёсель #{j}", end="")

	print()

	X = np.array(X).astype(np.float32)
	T = np.concatenate((T_cats, T_dogs), axis=0).astype(np.float32)
	return X, T


class CnnCatDogV4BN(Model):
	def __init__(self):
		super().__init__()
		self.layers = [
			# -> (N, 3, 112, 112)

			ZeroPad(), Conv(r=16, r_prev=3), BatchNorm(16), ReLU(),  # -> (N, 16, 112, 112)
			ZeroPad(), Conv(r=16, r_prev=16), BatchNorm(16), ReLU(),  # -> (N, 16, 112, 112)
			MaxPool(),  # -> (N, 16, 56, 56)

			ZeroPad(), Conv(r=32, r_prev=16), BatchNorm(32), ReLU(),  # -> (N, 32, 56, 56)
			ZeroPad(), Conv(r=32, r_prev=32), BatchNorm(32), ReLU(),  # -> (N, 32, 56, 56)
			MaxPool(),  # -> (N, 32, 28, 28)

			ZeroPad(), Conv(r=64, r_prev=32), BatchNorm(64), ReLU(),  # -> (N, 64, 28, 28)
			ZeroPad(), Conv(r=64, r_prev=64), BatchNorm(64), ReLU(),  # -> (N, 64, 28, 28)
			MaxPool(),  # -> (N, 64, 14, 14)

			ZeroPad(), Conv(r=128, r_prev=64), BatchNorm(128), ReLU(),  # -> (N, 128, 14, 14)
			ZeroPad(), Conv(r=128, r_prev=128), BatchNorm(128), ReLU(),  # -> (N, 128, 14, 14)
			MaxPool(),  # -> (N, 128, 7, 7)

			Flatten(),
			Dropout(),
			Dense(128 * 7 * 7, 256), BatchNorm(256), ReLU(),  # -> (N, 256)
			Dense(256, 2),  # -> (N, 2)
			SoftmaxCrossEntropy()  # -> (N, 2)
		]


def predict_batched(m, X, batch_size=16):
	preds = []
	print("calc metrics...")
	for i in range(0, len(X), batch_size):
		print(f"\r{i}", end="")

		X_batch = X[i: i + batch_size]
		# Важно: для predict выключаем Dropout и т.д.
		preds.append(m.predict(X_batch))
	print()
	return np.concatenate(preds, axis=0)


def augment_batch(X_batch: np.ndarray):
	# Работаем с копией, чтобы не испортить основной X_train
	X_aug = X_batch.copy()

	# 1. Горизонтальный флип (отзеркаливание)
	# Генерируем маску: для каких картинок в батче делаем флип
	flip_mask = np.random.rand(X_aug.shape[0]) > 0.5
	# В тензоре (N, C, H, W) ширина — это последняя ось (3)
	X_aug[flip_mask] = X_aug[flip_mask, :, :, ::-1]

	# 2. Можно добавить чуть-чуть случайного шума (необязательно)
	# noise = np.random.normal(0, 0.02, X_aug.shape).astype(np.float32)
	# X_aug += noise

	return X_aug


if __name__ == '__main__':
	# Hyperparams
	train_size = 10000
	val_size = 2000
	train_rat = round(train_size / (train_size + val_size), 2)
	epochs = 20
	batch_size = 32
	lr = 0.1
	save_step = 1
	ts = (64, 64)

	model = CnnCatDogV4BN()
	optim = Optimizer(model, lr)

	# Data load
	print("DATA LOADING...")
	X_train, T_train = load_dataset(0, train_size, ts)
	X_val, T_val = load_dataset(train_size, train_size + val_size, ts)
	print("DATA LOADED!")

	# Metrics
	training_losses = []
	validation_losses = []
	accuracies = []
	confusion_matrices = []

	# Train
	if not os.path.exists(model.__class__.__name__):
		os.makedirs(model.__class__.__name__)
	start_time = time.time()

	print(f"Hyperparams [alg]:\n"
	      f"train ratio = {train_rat},\n"
	      f"batch size = {batch_size},\n"
	      f"learning rate = {lr}\n")

	for epoch in range(epochs):
		# Перемешиваем датасет
		X_train, T_train = shuffle_dataset(X_train, T_train)

		# Batch iter
		iters = X_train.shape[0] // batch_size
		for i in range(iters):
			progress = (i + 1) / iters * 10
			progress_bar = int(progress) * "▐"
			# ▬
			print(f"\r{round((i + 1) / iters * 100, 2)}%  {progress_bar}", end="")
			X_batch = X_train[batch_size * i: batch_size * (i + 1)]
			T_batch = T_train[batch_size * i: batch_size * (i + 1)]

			# Forward
			X_batch_aug = augment_batch(X_batch)
			Y_batch = model.forward(X_batch_aug)

			# Backward
			optim.backward(T_batch)

			# Update params
			optim.step()

		# Logging & Save
		if epoch % save_step == 0:
			print("SAVE PARAMS ...")
			# Y_val = model.predict(X_val)
			# Y_train = model.predict(X_train)

			# Предсказания модели
			Y_val = predict_batched(model, X_val, batch_size=128)
			Y_train = predict_batched(model, X_train, batch_size=128)

			# Рассчитываем метрики
			cm = confusion_matrix(Y_val, T_val)
			accuracy = float(np.trace(cm) / np.sum(cm))
			train_loss = float(cross_entropy(Y_train, T_train))
			val_loss = float(cross_entropy(Y_val, T_val))
			confusion_matrices.append(cm)
			training_losses.append(train_loss)
			validation_losses.append(val_loss)
			accuracies.append(accuracy)
			log = (
				f"Epoch:            {epoch} / {epochs}\n"
				f"Time:             {round(time.time() - start_time, 2)} s.\n"
				f"Accuracy:         {round(accuracy * 100, 2)}%\n"
				f"Overfit:          {round(100 - train_loss / val_loss * 100, 2)}%\n"
				f"Training Loss:    {train_loss:.2e}\n"
				f"Validation Loss:  {val_loss:.2e}"
			)

			print()
			print(log)
			with open(f"{model.__class__.__name__}/EP_{epoch}_METRIX.txt", "w") as f:
				f.write(log + "\n" + "[CONFUSION MATRIX]\n" + str(cm))

			model.save_params(f"{model.__class__.__name__}/EP_{epoch}_PARAMS.npz")
			print("PARAMS SAVED!")
			print()

	# Visualization
	fig, axs = plt.subplots(1, 2, figsize=(10, 6))  # два графика в ряд
	axs = axs.flatten()

	eps = [ep * save_step for ep in range(epochs // save_step)]

	# === График ошибок ===
	axs[0].set_title("Error per epoch")
	axs[0].set_xlabel("Epoch Number")
	axs[0].set_ylabel("Error")
	axs[0].grid(True, linestyle='--', alpha=0.7)
	axs[0].scatter(eps, training_losses, s=5, color="#483D8B", label="train error")
	axs[0].scatter(eps, validation_losses, s=5, color="#CD5C5C", label="val error")
	axs[0].legend()

	# === График точности ===
	axs[1].set_title("Metrics")
	axs[1].set_xlabel("Epoch Number")
	axs[1].set_ylabel("Accuracy")
	axs[1].grid(True, linestyle='--', alpha=0.7)
	axs[1].set_ylim(0, 1)
	axs[1].scatter(eps, accuracies, s=5, color="#CD5C5C", label="accuracy")
	axs[1].scatter(eps, [1 - tr / val for tr, val in zip(training_losses, validation_losses)],
	               s=5, color="blue", label="overfit")
	axs[1].legend()

	# Текст
	plt.tight_layout()
	fig.text(
		x=0.01, y=0.2,
		s=f"Hyperparameters\n"
		  f"Learning Rate: {lr}\n"
		  f"Epoch Count: {epochs}\n"
		  f"Training Ratio: {train_rat}\n"
		  f"Batch Size: {batch_size}"
	)
	fig.text(
		x=0.66, y=0.2,
		s=f"Params\n"
		  f"{[l.params[0].size + l.params[1].size if l.params else 0 for l in model.layers]}"
	)
	t_tot = time.time() - start_time

	fig.text(
		x=0.33, y=0.2,
		s=f"Time\n"
		  f"Total: {round(t_tot, 2)} s.\n"
		  f"Per epoch: {round(t_tot / epochs, 2)}"
	)

	plt.subplots_adjust(bottom=0.5)
	# Открытие графика на весь экран
	manager = plt.get_current_fig_manager()
	manager.window.state('zoomed')

	# Сохранение картинки
	plt.savefig(f"{model.__class__.__name__}/TRAIN.png", dpi=300, bbox_inches="tight")
	plt.show()
