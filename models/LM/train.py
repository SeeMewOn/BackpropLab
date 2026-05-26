import time

from NN.functions import get_combined_mask
from NN.optimizer import Adam
from models.LM.language_model import LanguageModel
from utils.backend import np


def loss(Y, T, padding_mask=None) -> float:
	"""
	Вычисляет скалярный Loss
	T: (B, L) - индексы правильных токенов
	"""
	B, L, V = Y.shape
	# Используем продвинутую индексацию numpy/cupy, чтобы достать
	# только нужные вероятности.
	batch_idx = np.arange(B)[:, None]  # (B, 1)
	seq_idx = np.arange(L)[None, :]  # (1, L)

	# Выбираем вероятности правильных классов
	probs = Y[batch_idx, seq_idx, T]

	if padding_mask is not None:
		probs *= padding_mask
		loss = -np.sum(np.log(probs + 1e-10)) / np.sum(padding_mask)
		return float(loss)

if __name__ == '__main__':
	epochs = 1
	logging_step = 5  # Раз во сколько шагов сохранять параметры и считать метрики
	batch_size = 8
	val_total_ratio = 0.2
	vocab_size = 16000
	d_model = 256
	context_size = 512

	model = LanguageModel(vocab_size, d_model, context_size)
	model.train()
	optimizer = Adam(model.get_params())

	# TODO загрузка реального датасета текстов + контролируемое рандомное перемешивание
	# Dataset load
	X_full = np.random.randint(low=0, high=vocab_size, size=(1_000_000, context_size)) # (Total, L)
	pad_token_id = 0

	val_size = int(X_full.shape[0] * val_total_ratio)
	batch_iters = X_full.shape[0] // batch_size

	# splitting the dataset into training and validation
	X_train = X_full[val_size:]
	X_val = X_full[:val_size]

	for epoch in range(epochs):

		# TODO Shuffle dataset

		start = time.time()

		for i in range(batch_iters):
			# Batch (B, L)
			# print("Создание батча ...")
			X = X_train[batch_size * i: batch_size * (i + 1)]   # (B, L)
# 			print("Батч создан!")

			# Targets (B, L) TODO Реальные метки
			T = np.random.randint(low=0, high=vocab_size, size=(batch_size, context_size))

			# TODO padding_mask использовать только на инференсе.
			#  Подготовить данные так, чтобы текст не нуждался в заполнении PAD-ами
# 			print("Создание масок ...")
			# Combined mask. 1 - выше главной диагонали и на месте PAD токенов. 0 - содержательные токены
			mask = get_combined_mask(X, pad_token_id)  # (B, 1, L, L)
			# Padding mask. 1 - содержательные токены
			padding_mask = X != pad_token_id
# 			print("Маски созданы!")

			# FORWARD
# 			print("Расчёт предсказаний...")
			Y = model.forward(X, mask)
# 			print("Предсказания рассчитаны!")

			# BACKWARD
# 			print("Расчёт градиентов...")
			model.backward(T, None)
			# print("Градиенты рассчитаны!")

			# STEP
# 			print("Обновление градиентов ...")
			optimizer.step()
			model.zero_grad()
# 			print("Градиенты обновлены!")

			# Logging
			t = optimizer.t
			print(f"Step {t} / {batch_iters}, {round(time.time() - start, 2)} sec")
			# np.get_default_memory_pool().free_all_blocks()