import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CausalLMDataset(Dataset):
	def __init__(self, bin_path, context_size=512, seed=42):
		rng = np.random.default_rng(seed=seed)
		self.context_size = context_size

		# Подключение к файлу
		self.tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')

		# Считаем, сколько полных блоков по (L + 1) токенов помещается
		self.block_size = context_size + 1
		self.num_samples = len(self.tokens) // self.block_size

		# Изначально массив индексов статический [0, 1, 2, ..., N-1]
		self.indices = np.arange(self.num_samples)
		rng.shuffle(self.indices)

		print(f"[{bin_path}] Итого семплов в базе: {self.num_samples}")

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		# Мапим виртуальный idx из DataLoader в наш детерминированно перемешанный active_indices
		actual_block_idx = self.indices[idx]

		start_idx = actual_block_idx * self.block_size
		end_idx = start_idx + self.block_size
		chunk = self.tokens[start_idx:end_idx]

		x = torch.tensor(chunk[:-1], dtype=torch.long)
		t = torch.tensor(chunk[1:], dtype=torch.long)

		return x, t
