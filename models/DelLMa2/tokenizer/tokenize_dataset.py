import os
import glob
import time

import numpy as np
from pathlib import Path
from tokenizers import Tokenizer


def tokenize_shards_to_bin(input_dir, output_bin_path, tokenizer_path, text_end_tag="<endoftext>"):
	"""
	Чтение шардов из input_dir, токенизация, сохранение их в единый .bin файл.
	Используем uint16, так как размер словаря 16000 (умещается в диапазон 0-65535)
	TODO проверка размера словаря
	"""
	tokenizer = Tokenizer.from_file(tokenizer_path)

	path = Path(input_dir)
	shard_files = sorted(list(path.glob("*.txt")))

	print(f"=== Старт бинарной токенизации ===")
	print(f"Из папки: {input_dir}")
	print(f"В файл:   {output_bin_path}")

	batch = []
	buffer_ids = []
	total_tokens = 0

	# Лимит токенов в памяти перед сбросом на диск
	FLUSH_THRESHOLD = 5_000_000
	MAX_BATCH_SIZE = 2048

	start_time = time.time()
	with open(output_bin_path, "wb") as f_bin:
		for shard_file in shard_files:
			print(f"Обработка шарда: {os.path.basename(shard_file)}")
			print()
			if not os.path.isfile(shard_file):
				continue

			with open(shard_file, "r", encoding="utf-8") as f_shard:
				block = []
				for line in f_shard:
					# Как только дошли до метки конца текста - сборка block в строку
					if line.strip() == text_end_tag:
						block_str = "".join(block) + "<|endoftext|>"
						batch.append(block_str)
						block = []

						# Как только батч превысил размер MAX_BATCH_SIZE текстов - токенизация
						if len(batch) >= MAX_BATCH_SIZE:
							encodings = tokenizer.encode_batch(batch)
							for encoding in encodings:
								buffer_ids.extend(encoding.ids)
							batch = []

							# Как только размер массива токенов превысил FLUSH_THRESHOLD - запись в файл
							if len(buffer_ids) >= FLUSH_THRESHOLD:
								arr = np.array(buffer_ids, dtype=np.uint16)
								f_bin.write(arr.tobytes())
								total_tokens += len(arr)
								buffer_ids = []

								t = time.time() - start_time
								print(
									f"\rTotal tokens: {total_tokens} | "
									f"Time: {t:.2f} s | "
									f"Speed: {total_tokens / t:.2f} s/",
									end="",
									flush=True
								)
					else:
						# Накопление одного блока (книга либо вики-статья)
						block.append(line)

		# Дотокензируем остатки, которые не вошли в последний батч
		if batch:
			encodings = tokenizer.encode_batch(batch)
			for e in encodings:
				buffer_ids.extend(e.ids)

		# Сбрасываем финальные остатки буфера на диск
		if buffer_ids:
			arr = np.array(buffer_ids, dtype=np.uint16)
			f_bin.write(arr.tobytes())
			total_tokens += len(arr)

	print(f"=== Готово! ===")
	print(f"Всего токенов записано: {total_tokens:,}")
	print(f"Итоговый размер бинарного файла: {os.path.getsize(output_bin_path) / (1024 ** 2):.2f} MB\n")


def verify_binary_dataset(bin_path, tokenizer_path, num_tokens_to_show=400):
	"""
	Проверка чтения из бинарного файла
	"""
	if not os.path.exists(bin_path):
		print(f"❌ Ошибка: Файл {bin_path} не найден!")
		return
	if not os.path.exists(tokenizer_path):
		print(f"❌ Ошибка: Токенизатор {tokenizer_path} не найден!")
		return

	# 1. Загружаем токенизатор
	tokenizer = Tokenizer.from_file(tokenizer_path)

	# 2. Подключаемся к бинарнику через memmap (строго uint16, как при записи)
	tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
	total_tokens = len(tokens)

	print(f"=== Проверка датасета ===")
	print(f"Файл:         {bin_path}")
	print(f"Всего токенов: {total_tokens:,}")
	print(f"Размер словаря токенизатора: {tokenizer.get_vocab_size():,}")
	print(f"=========================\n")

	if total_tokens == 0:
		print("❌ Файл пустой, декодировать нечего.")
		return

	# Проверка №1: Кусок из самого начала файла
	print(f"--- [Тест 1] Первые {num_tokens_to_show} токенов ---")
	sample_start = tokens[:num_tokens_to_show].tolist()
	text_start = tokenizer.decode(sample_start)
	print(text_start)
	print("\n" + "=" * 50 + "\n")

	# Проверка №2: Кусок из случайного места (проверить, что в середине нет каши)
	if total_tokens > num_tokens_to_show * 2:
		print(f"--- [Тест 2] Случайный кусок из середины датасета ---")
		# Генерируем случайный валидный индекс старта
		random_start = np.random.randint(0, total_tokens - num_tokens_to_show)
		sample_middle = tokens[random_start: random_start + num_tokens_to_show].tolist()
		text_middle = tokenizer.decode(sample_middle)
		print(text_middle)
		print("\n" + "=" * 50 + "\n")

	# Микро-валидация на границы словаря
	max_token_id = np.max(tokens)
	vocab_size = tokenizer.get_vocab_size()
	print(f"Проверка индексов: максимальный ID токена в файле = {max_token_id}")
	if max_token_id >= vocab_size:
		print(
			f"❌ ВНИМАНИЕ: Найдено критическое смещение! Токен {max_token_id} выходит за рамки словаря ({vocab_size}).")
	else:
		print("✅ Индексы в норме, оут-оф-вокаб токенов не обнаружено.")


if __name__ == '__main__':
	# tokenize_shards_to_bin(
	# 	input_dir="../../../data/DELLMA_2_DATA/validation",
	# 	output_bin_path="../../../data/DELLMA_2_DATA/binary/val.bin",
	# 	tokenizer_path="saved_tokenizer/tokenizer_delLMa2.json"
	# )

	verify_binary_dataset(
		bin_path="../../../data/DELLMA_2_DATA/binary/val.bin",
		tokenizer_path="saved_tokenizer/tokenizer_delLMa2.json",
		num_tokens_to_show=300
	)