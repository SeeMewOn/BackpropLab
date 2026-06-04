
import os
from tokenizers import Tokenizer, pre_tokenizers
TOK_JSON = "saved_tokenizer/tokenizer_delLMa2.json"

def test():
	tokenizer = Tokenizer.from_file(TOK_JSON)
	# --- ГЛУБОКАЯ ПРОВЕРКА РАБОТЫ ---
	print("\n--- Проверка токенизации и изоляции \\n ---")

	# Тестовый текст со сложной структурой переносов и пробелов
	test_text = "<|user|>\nПривет!\nКак твои дела?\n\n    Я проверяю отступы.\n<|assistant|>\nВсе отлично!<|endoftext|>"

	encoded = tokenizer.encode(test_text)
	decoded_text = tokenizer.decode(encoded.ids)

	print(f"\n[Исходный текст]:\n{test_text}\n")
	print(f"[Токены (IDs)]:\n{encoded.ids}\n")
	print(f"[Строковые токены]:\n{encoded.tokens}\n")
	print(f"[Декодированный текст]:\n{decoded_text}\n")

	# Стресс-тест на склеивание: проверяем, что '\n' имеет свой уникальный чистый ID
	newline_byte_repr = tokenizer.pre_tokenizer.pre_tokenize_str("\n")[0][0]
	newline_id = tokenizer.token_to_id(newline_byte_repr)
	print(f"Эталонный ID для одиночного переноса строки '\\n': {newline_id}")

	# Проверяем, нет ли в словаре запрещенных склеек
	has_merged_newlines = False
	for token_str, token_id in tokenizer.get_vocab().items():
		# 'Ċ' — это представление символа '\n' в ByteLevel пространстве
		if "Ċ" in token_str and token_str != "Ċ":
			# Если нашли токен, где Ċ стоит рядом с буквами или пробелами
			print(f"ВНИМАНИЕ: Найден склеенный токен: {token_str} -> ID: {token_id}")
			has_merged_newlines = True

	if not has_merged_newlines:
		print("БРАВО! Проверка пройдена: ни один перенос строки не склеился с текстом или пробелами!")


def get_hf_char_to_byte():
	"""
	Воссоздает правильную карту маппинга символов обратно в байты,
	которую Hugging Face ByteLevel использует под капотом (стандарт GPT-2).
	"""
	bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
	cs = bs[:]
	n = 0
	for b in range(2 ** 8):
		if b not in bs:
			bs.append(b)
			cs.append(2 ** 8 + n)
			n += 1

	# Прямой маппинг: байт -> символ BPE
	byte_to_char = {b: chr(c) for b, c in zip(bs, cs)}

	# Обратный маппинг: символ BPE -> физический байт
	return {v: k for k, v in byte_to_char.items()}


def audit_tokenizer(tokenizer_json_path: str, output_txt_path: str):
	if not os.path.exists(tokenizer_json_path):
		raise FileNotFoundError(f"Не найден файл токенизатора: {tokenizer_json_path}")

	tokenizer = Tokenizer.from_file(tokenizer_json_path)
	vocab = tokenizer.get_vocab()
	sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

	# Получаем ОРИГИНАЛЬНУЮ, не сломанную карту символов
	char_to_byte = get_hf_char_to_byte()

	total_bytes = 0
	vocab_size = len(sorted_vocab)

	print(f"Загружен словарь размером: {vocab_size} токенов.")
	print("Декодирую BPE-символы в нормальный текст...")

	with open(output_txt_path, "w", encoding="utf-8") as f:
		f.write(f"=== СЛОВАРЬ DELLMA-2 ===\n")
		f.write(f"Размер словаря: {vocab_size} токенов\n")
		f.write("-" * 90 + "\n")
		f.write(f"{'ID':<7} | {'Вес (байт)':<10} | {'Сырой BPE токен':<25} | {'Реальный текст'}\n")
		f.write("-" * 90 + "\n")

		for token_str, token_id in sorted_vocab:
			try:
				# Переводим каждый символ BPE-строки в реальный физический байт
				token_bytes = bytes([char_to_byte[c] for c in token_str])

				# Пробуем декодировать байты в нормальный UTF-8 текст
				decoded_text = token_bytes.decode("utf-8")
				display_text = f'"{decoded_text}"'
			except KeyError:
				# Для спецтокенов (<|endoftext|>, <|user|> и т.д.), которых нет в ByteLevel
				token_bytes = token_str.encode("utf-8")
				display_text = f'[SPECIAL] {token_str}'
			except UnicodeDecodeError:
				# Если токен — это обрубок (часть мультибайтового символа кириллицы),
				# он не может собраться в UTF-8 один. Выводим его замену + HEX.
				decoded_text = token_bytes.decode("utf-8", errors="replace")
				display_text = f'"{decoded_text}" [Часть символа / Hex: 0x{token_bytes.hex().upper()}]'

			byte_len = len(token_bytes)
			total_bytes += byte_len

			# Экранируем переносы, чтобы не ломать строки в txt файле
			visible_raw = token_str.replace("\n", "\\n").replace("\t", "\\t")
			visible_display = display_text.replace("\n", "\\n").replace("\t", "\\t")

			f.write(f"{token_id:<7} | {byte_len:<10} | {visible_raw:<25} | {visible_display}\n")

	avg_bytes_per_token = total_bytes / vocab_size

	print("\n" + "=" * 45)
	print(f"АНАЛИЗ ЗАВЕРШЕН УСПЕШНО:")
	print(f"Результат сохранен в: {output_txt_path}")
	print(f"Общий вес всех байт словаря: {total_bytes} байт")
	print(f"Средний вес одного токена: {avg_bytes_per_token:.2f} байт")
	print("=" * 45)


if __name__ == "__main__":
	audit_tokenizer(TOK_JSON, "readable_dellma_2_vocab.txt")
	audit_tokenizer("saved_tokenizer/tokenizer_wiki.json", "readable_dellma_1_vocab.txt")
