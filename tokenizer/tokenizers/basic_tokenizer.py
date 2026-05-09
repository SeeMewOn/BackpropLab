from tokenizer.tokenizers.tokenizer import Tokenizer, get_freqs, merge


class BasicTokenizer(Tokenizer):
	def __init__(self):
		super().__init__()

	def train(self, text, vocab_size, verbose=False):
		assert vocab_size >= 256
		num_merges = vocab_size - 256

		# input text preprocessing
		text_bytes = text.encode("utf-8")
		ids = list(text_bytes)  # list of int in range 0..255

		merges = {}  # {(int, int): int, ...}
		vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
		for i in range(num_merges):
			# Находим самую частую пару
			freqs = get_freqs(ids)
			pair = max(freqs, key=freqs.get)

			# Назначаем новому токену следующий свобойдный id,
			# заменяем самые частые пары на новый токен
			idx = 256 + i
			ids = merge(ids, pair, idx)

			# Сохраняем объединение
			merges[pair] = idx
			vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

			if verbose:
				print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]} has {freqs[pair]} occurrences)")

		self.merges = merges
		self.vocab = vocab

	def decode(self, ids):
		# given ids (list of integers), return Python string
		text_bytes = b"".join([self.vocab[idx] for idx in ids])
		text = text_bytes.decode("utf-8", errors="replace")
		return text

	def encode(self, text):
		# input text preprocessing
		text_bytes = text.encode("utf-8")
		ids = list(text_bytes)  # list of int in range 0..255

		while len(ids) >= 2:
			freqs = get_freqs(ids)
			# Ищем пару с минимальным индексом слияния,
			# т.е. пару которая раньше стоит в merges.
			# Строка self.merges.get(p, float("inf"))
			# означает "если я не знаю эту пару, она
			# максимально бесполезна". Мы даём ей
			# бесконечный id, чтобы функция min никогда
			# ее не выбрала, пока в тексте есть хотя бы
			# одна пара, которую мы знаем
			pair = min(freqs, key=lambda p: self.merges.get(p, float("inf")))
			if pair not in self.merges:
				break

			idx = self.merges[pair]
			ids = merge(ids, pair, idx)

		return ids


if __name__ == '__main__':
	d = {
		(1, 2): 2,
		(3, 1): 1,
		(2, 3): 1,

	}
	print(max(d))
