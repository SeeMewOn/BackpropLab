from NN.functions import get_combined_mask
from models.LM.language_model import LanguageModel
from utils.backend import np




if __name__ == '__main__':
	epochs = 10
	batch_size = 8
	val_total_ratio = 0.2
	vocab_size = 16000
	d_model = 512
	context_size = 1024
	model = LanguageModel(vocab_size, d_model, context_size)

	# TODO загрузка реального датасета текстов + контролируемое рандомное перемешивание
	# Dataset load
	X_full = np.random.randint(low=0, high=vocab_size, size=(1_000_000, context_size)) # (Total, L)
	# T_full = np.random.randint(low=0, high=vocab_size, size=(1_000_000, context_size)) # (Total, L)
	pad_token_id = 0

	val_size = int(X_full.shape[0] * val_total_ratio)
	batch_iters = X_full.shape[0] // batch_size

	# splitting the dataset into training and validation
	X_train = X_full[val_size:]
	X_val = X_full[:val_size]

	for epoch in range(epochs):

		# Shuffle dataset

		for i in range(batch_iters):
			# Batch (B, L)
			X = X_train[batch_size * i: batch_size * (i + 1)]

			# TODO не вычислять padding mask 2 раза.
			#  Вычислить один раз и инвертировать.
			# Combined mask. 1 - выше главной диагонали и на месте PAD токенов. 0 - содержательные токены
			mask = get_combined_mask(X, pad_token_id)  # (B, 1, L, L)
			# Padding mask. 1 - содержательные токены
			padding_mask = X != pad_token_id

			# FORWARD
			model.train()
			Y = model.forward(X, mask)

			# BACKWARD
			# model.backward(T, padding_mask)

			# STEP
			# TODO optimizer.step()
			model.zero_grad()
