class ShardWriter:
	"""
	Класс ShardWriter отвечает за запись текста
	в шарды - файлы, размер которых ограничен сверху.
	"""
	def __init__(
			self,
			shard_dir,
			shard_pref,
			max_shard_size_gb=1.0,
	):
		"""
		:param shard_dir: Директория, в которую пишутся шарды.
		:param shard_pref: Название шардов. Через '_' после названия стоит номер шарда.
		:param max_shard_size_gb: Максимальный размер шарда в гигабайтах. По умолчанию равен 1 GB.
		"""
		self.shard_dir = shard_dir
		self.shard_pref = shard_pref
		self.max_bytes = max_shard_size_gb * 1024 * 1024 * 1024
		self.shard_index = 1
		self.current_shard_bytes = 0
		self.f_current = open(self._get_shard_path_by_index(self.shard_index), "w", encoding="utf-8")
		self.saved_count = 0

	def _get_shard_path_by_index(self, shard_index):
		return f"{self.shard_dir}/{self.shard_pref}_{shard_index}.txt"

	def write_doc(self, block: str):
		block_bytes = len(block.encode('utf-8'))
		# Проверяем, не переполнился ли текущий файл
		if self.current_shard_bytes + block_bytes > self.max_bytes:
			self.flush()
			print(
				f"Сформирован шард {self.shard_pref}_{self.shard_index}: {self.current_shard_bytes / (1024 ** 2):.2f} МБ.")

			# Открываем новый файл
			self.shard_index += 1
			self.f_current = open(self._get_shard_path_by_index(self.shard_index), "w", encoding="utf-8")
			self.current_shard_bytes = 0

		# Пишем блок в текущий шард
		self.f_current.write(block)
		self.current_shard_bytes += block_bytes
		self.saved_count += 1

	def flush(self):
		self.f_current.flush()
		self.f_current.close()
