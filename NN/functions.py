from PIL import Image

from deprecated.enums import RegMode
from utils.backend import np


def softmax(X):
	"""
	Универсальный softmax для тензоров любой размерности
	"""
	X_exp = np.exp(X - np.max(X, axis=-1, keepdims=True))
	sum_exp = np.sum(X_exp, axis=-1, keepdims=True)
	return X_exp / sum_exp


def get_combined_mask(X, pad_token_id=0):
	"""
	X_indices: (B, L) - токены
	"""
	B, L = X.shape

	# 1. Causal Mask: (L, L)
	# Создаем матрицу, где выше главной диагонали стоят 1.
	# Это "запрет смотреть вперёд"
	causal_mask = np.triu(np.ones((L, L)), k=1).astype(np.bool_)

	# 2. Padding Mask: (B, 1, 1, L)
	# Помечаем True там, где стоит PAD токен
	padding_mask = (X == pad_token_id)
	padding_mask = padding_mask[:, np.newaxis, np.newaxis, :]

	# 3. Объединяем их через логическое ИЛИ
	# Если токен "в будущем" ИЛИ он "паддинг" -> маскируем (True)
	combined_mask = causal_mask | padding_mask

	return combined_mask  # (B, 1, L, L)

def im2tensor(img_path, target_size=(64, 64), augment=False):
	img = Image.open(img_path).convert("RGB")

	if augment:
		pass

	# Crop
	W, H = img.size
	crop_size = min(H, W)
	left = (W - crop_size) / 2
	top = (H - crop_size) / 2
	right = (W + crop_size) / 2
	bottom = (H + crop_size) / 2
	img = img.crop((left, top, right, bottom))
	img = img.resize(target_size, Image.Resampling.BILINEAR)
	img_tensor = np.array(img).astype(np.float32)
	img_tensor = img_tensor.transpose(2, 0, 1) / 255.0 - 0.5  # (H, W, C) -> (C, H, W)
	return img_tensor


def shuffle_dataset(x: np.ndarray, t: np.ndarray) -> (np.ndarray, np.ndarray):
	indices = np.random.permutation(x.shape[0])
	x = x[indices]
	t = t[indices]
	return x, t


def cross_entropy(Y: np.ndarray, T: np.ndarray):
	Y = np.clip(Y, 1e-12, 1 - 1e-12)
	res = - np.sum(np.multiply(T, np.log(Y))) / Y.shape[0]
	return res


def confusion_matrix(
		Y: np.ndarray,
		T: np.ndarray,
):
	K = T.shape[1]
	# Y = softmax(X @ W.T + b)
	predicted = np.argmax(Y, axis=1)  # (N,)
	true = np.argmax(T, axis=1)  # (N,)
	cm = np.zeros((K, K), dtype=np.int32)
	np.add.at(cm, (true, predicted), 1)
	return cm


def phi_polynom(x, degree: int) -> np.ndarray:
	"""
	Функция фи.
	:param degree: Степень.
	:param x: Input Variable.
	:return: Вектор (1, x, x^2, ..., x^degree).
	"""
	return np.array([(x ** i) for i in range(degree + 1)])


def error(
		w: np.ndarray,
		T: np.ndarray,
		Phi: np.ndarray,
		mode: RegMode = RegMode.LEAST_SQUARE_REGRESSION,
		l: float | None = None,
) -> float:
	"""
	Ошибка.
	:param l:
	:param mode:
	:param w: Вектор весов (размер M).
	:param T: Вектор Target Variables (размер N).
	:param Phi: Design Matrix (размер N x M).
	:return: Error.
	"""
	predictions = Phi @ w  # (N,)
	residuals = predictions - T  # (N,)
	base_e = 0.5 * np.sum(residuals ** 2)

	if mode == RegMode.LEAST_SQUARE_REGRESSION:
		return base_e
	elif mode == RegMode.RIDGE_REGRESSION:
		return base_e + 0.5 * l * np.sum(w[1:]) ** 2
	elif mode == RegMode.LASSO_REGRESSION:
		return base_e + 0.5 * l * np.sum(np.abs(w)[1:])


# def softmax_vector(z: np.ndarray):
# 	"""
# 	Стандартная функция Softmax.
# 	Вычитание максимальной компоненты из вектора z даёт численную стабильность.
# 	:param z: Вектор.
# 	:return: Вектор.
# 	"""
# 	z_exp = np.exp(z - np.max(z))
# 	y = z_exp / np.sum(z_exp)
# 	return y
#
#
# def softmax(Z: np.ndarray) -> np.ndarray:
# 	"""
# 	Матричная функция Softmax. Может принимать на вход как вектор, так и матрицу.
#
# 	- В случае вектора возвращает стандартный вектор Softmax;
# 	- В случае матрицы возвращает матрицу, строки которой - Softmax-ы соответствующих строк входной матрицы.
#
# 	Вычитание максимальной компоненты из вектора z даёт численную стабильность.
# 	:param Z: Вектор или матрица.
# 	:return: Вектор или матрица Softmax.
# 	"""
# 	Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
# 	S = np.sum(Z_exp, axis=1, keepdims=True)
# 	return Z_exp / S


# def grad_log_reg_vec(W, b, X, T):
#     Y = softmax(X @ W.T + b)         # (N×K)
#     delta = Y - T                    # (N×K)
#     grad_W = delta.T @ X / X.shape[0]  # (K×M)
#     grad_b = delta.mean(axis=0)        # (K,)
#     return np.concatenate((grad_W.flatten(), grad_b))

def grad_lin_reg(
		w: np.ndarray,
		T: np.ndarray,
		Phi: np.ndarray,
		mode: RegMode = RegMode.LEAST_SQUARE_REGRESSION,
		l: float | None = None
) -> np.ndarray:
	base_grad = Phi.T @ (Phi @ w - T)
	if mode == RegMode.LEAST_SQUARE_REGRESSION:
		return base_grad
	elif mode == RegMode.RIDGE_REGRESSION:
		mask = np.ones_like(w)
		mask[0] = 0
		return base_grad + l * (w * mask)
	elif mode == RegMode.LASSO_REGRESSION:
		mask = np.ones_like(w)
		mask[0] = 0
		return base_grad + l * (np.sign(w) * mask)


def error_log_reg(
		W: np.ndarray,
		b: np.ndarray,
		X: np.ndarray,
		T: np.ndarray,
):
	N = X.shape[0]
	err = np.float32(0)
	for i in range(N):
		x = X[i]  # (M,)
		t = T[i]  # (K,)
		class_index = np.argmax(t)
		z = W @ x + b  # (K,)
		# z_max = np.max(z)
		err += -z[class_index] + np.log(np.sum(np.exp(z)))

	return err / N


def grad_log_reg(
		W: np.ndarray,
		b: np.ndarray,
		X: np.ndarray,
		T: np.ndarray,
):
	"""
	Градиент для логистической регрессии
	:param W: Матрица весов (K×M).
	:param b: Вектор bias-ов (K).
	:param X: Матрица признаков (N×M).
	:param T: Матрица строк one-hot encoding векторов (N×K).
	:return: Не 1/N * gradient, а кортеж (grad_W / N, grad_b / N).
	"""
	N = X.shape[0]
	# identity_vector = np.full(N, 1, dtype=np.float32)
	# B = np.outer(identity_vector, b)
	sm = softmax(X @ W.T + b) - T
	grad_b = sm.mean(axis=0)
	grad_W = sm.T @ X / N
	return grad_W, grad_b


# def get_confusion_matrix(
# 		X: np.ndarray,
# 		T: np.ndarray,
# 		W: np.ndarray,
# 		b: np.ndarray,
# ):
# 	K = T.shape[1]
# 	Y = softmax(X @ W.T + b)
# 	predicted = np.argmax(Y, axis=1)  # (N,)
# 	true = np.argmax(T, axis=1)  # (N,)
# 	cm = np.zeros((K, K), dtype=int)
# 	np.add.at(cm, (true, predicted), 1)
# 	return cm


def get_binary_classification_metrics(
		X: np.ndarray,
		T: np.ndarray,
		W: np.ndarray,
		b: np.ndarray,
		tau: np.float32,
):
	# Вероятности для класса 1
	probs_class1 = softmax(X @ W.T + b)[:, 1]  # (N,)

	# Предсказания и истинные классы
	predicted = (probs_class1 >= tau).astype(int)  # (N,)
	true = np.argmax(T, axis=1)  # (N,)

	# Вычисляем TP, TN, FP, FN векторно
	TP = np.sum((predicted == 1) & (true == 1))
	TN = np.sum((predicted == 0) & (true == 0))
	FP = np.sum((predicted == 1) & (true == 0))
	FN = np.sum((predicted == 0) & (true == 1))

	# Метрики
	N = len(T)
	accuracy = (TP + TN) / N
	alpha = FP / (FP + TN) if (FP + TN) != 0 else 0.0  # FPR
	beta = FN / (FN + TP) if (FN + TP) != 0 else 0.0  # FNR
	precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
	recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0

	return accuracy, alpha, beta, precision, recall
