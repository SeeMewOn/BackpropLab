import regex as re

from tokenizer.tokenizers.tokenizer import Tokenizer, get_freqs, merge

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
	def __init__(self, pattern=None):
		super().__init__()
		self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
		self.compiled_pattern = re.compile(self.pattern)
		self.special_tokens = {}
		self.inverse_special_tokens = {}

	def train(self, text, vocab_size, verbose=False):
		assert vocab_size >= 256
		num_merges = vocab_size - 256

		# Разделяем текст на чанки по нашему паттерну.
		# В отличие от базовой версии, где мы обучались на
		# одном гигантском списке ID, здесь мы работаем со списком списков.

		text_chunks = re.findall(self.compiled_pattern, text)

		# input text preprocessing
		ids = [list(ch.encode("utf-8")) for ch in text_chunks]  # list[list[bytes]]. Это list[chunk_ids]

		# Когда мы считаем статистику (get_freqs) и делаем слияние (merge),
		# мы делаем это внутри каждого куска отдельно.
		#
		# Это предотвращает слияние через границы! Пара никогда не будет найдена,
		# если один символ в конце одного слова, а другой — в начале
		# следующего (если регулярка их разделила).
		#
		# Это делает токены более семантически чистыми: пробел обычно
		# приклеивается к началу слова, но само слово не «срастается»
		# с предыдущей точкой или числом.

		merges = {}  # {(int, int): int, ...} - история слияний
		vocab = {idx: bytes([idx]) for idx in range(256)}
		for i in range(num_merges):
			freqs = {}
			for chunk_ids in ids:
				get_freqs(chunk_ids, freqs)

			pair = max(freqs, key=freqs.get)
			idx = 256 + i
			ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

			# Сохраняем объединение
			merges[pair] = idx
			vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

			if verbose:
				print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]} has {freqs[pair]} occurrences)")

		self.merges = merges
		self.vocab = vocab

	def register_special_tokens(self, special_tokens):
		# special_tokens is a dictionary of str -> int
		# example: {"<|endoftext|>": 100257}
		self.special_tokens = special_tokens
		self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

	def decode(self, ids):
		# given ids (list of integers), return Python string
		part_bytes = []
		for idx in ids:
			if idx in self.vocab:
				part_bytes.append(self.vocab[idx])
			elif idx in self.inverse_special_tokens:
				part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
			else:
				raise ValueError(f"invalid token id: {idx}")
		text_bytes = b"".join(part_bytes)
		text = text_bytes.decode("utf-8", errors="replace")
		return text

	def _encode_chunk(self, text_bytes):
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

	def encode_ordinary(self, text):
		"""Encoding that ignores any special tokens."""
		# split text into chunks of text by categories defined in regex pattern
		text_chunks = re.findall(self.compiled_pattern, text)
		# all chunks of text are encoded separately, then results are joined
		ids = []
		for chunk in text_chunks:
			chunk_bytes = chunk.encode("utf-8")  # raw bytes
			chunk_ids = self._encode_chunk(chunk_bytes)
			ids.extend(chunk_ids)
		return ids

	def encode(self, text, allowed_special="none_raise"):
		"""
		Unlike encode_ordinary, this function handles special tokens.
		allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
		if none_raise, then an error is raised if any special token is encountered in text
		this is the default tiktoken behavior right now as well
		any other behavior is either annoying, or a major footgun
		"""
		# decode the user desire w.r.t. handling of special tokens
		special = None
		if allowed_special == "all":
			special = self.special_tokens
		elif allowed_special == "none":
			special = {}
		elif allowed_special == "none_raise":
			special = {}
			assert all(token not in text for token in self.special_tokens)
		elif isinstance(allowed_special, set):
			special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
		else:
			raise ValueError(f"allowed_special={allowed_special} not understood")
		if not special:
			# shortcut: if no special tokens, just use the ordinary encoding
			return self.encode_ordinary(text)
		# otherwise, we have to be careful with potential special tokens in text
		# we handle special tokens by splitting the text
		# based on the occurrence of any exact match with any of the special tokens
		# we can use re.split for this. note that surrounding the pattern with ()
		# makes it into a capturing group, so the special tokens will be included
		special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
		special_chunks = re.split(special_pattern, text)
		# now all the special characters are separated from the rest of the text
		# all chunks of text are encoded separately, then results are joined
		ids = []
		for part in special_chunks:
			if part in special:
				# this is a special token, encode it separately as a special case
				ids.append(special[part])
			else:
				# this is an ordinary sequence, encode it normally
				ids.extend(self.encode_ordinary(part))
		return ids
