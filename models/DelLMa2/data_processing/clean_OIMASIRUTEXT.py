import re
import time
import os
import hashlib
from enum import Enum, auto

import unicodedata

# Dataset: https://huggingface.co/datasets/Oimasi/OIMASIRUTEXT
#
# Вход: OIMASIRUTEXT -> Выход: .txt файл с очищенным текстом
#
# Файлы датасета:
# merged_data_1.txt
# merged_data_2.txt
# merged_data_3.txt
# merged_data_4.txt
# merged_data_5.txt
# merged_data_6.txt
# merged_data_7.txt
#
# Каждый файл имеет структуру
# <startoftext>
# ...
# <endoftext>
# <startoftext>
# ...
# <endoftext>
# ...

DS_DIR = "../../../data/OIMASIRUTEXT"
CLEANED_DIR = "../../../data/CLEANED_OIMASIRUTEXT"

# === REGEX ПАТТЕРНЫ ДЛЯ ФИЛЬТРАЦИИ ===

RE_COMPLETED = re.compile(r'[.!?…]["»]?$')  # Строка не завершена
RE_META = re.compile(
	r"[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9._-]+|"  # Электронная почта
	r"Сконвертировано и опубликовано на|"
	r"Текст предоставлен ООО|"
	r"ЛитРес|"
	r"Fb2\.zipEpub|"
	r"copyright|"
	r"^популярность:[\s\t]+\*{0,2}?\d+|"
	r"Royallib|"
	r"^Оставить отзыв о книге:|"
	r"^Эта же книга в других форматах:|"
	r"^Все книги автора:|"
	r"\.txt|"
	r"\.htm|"
	r"\.jpg|"
	r"\.gif|"
	r"\[Электронный ресурс]|"
	r"^\d+ (\.\.)?(/.*)*$",  # 4 /dir/dir.ext
	re.IGNORECASE
)
TRASH_BOOKS = [
	# 2
	"Андрей Богатырев. Хрестоматия по программированию на Си в Unix", "ВИДЕО-94",
	"История морской культуры Южно-Китайского моря"
]
RE_TRASH_SECTIONS = re.compile(
	r"^литература\.?:?$|"
	r"^REFERENCES\.?:?$|"
	r"^содержание\.?:?$|"
	r"^примечания\.?:?$|"
	r"^оглавление\.?:?$|"
	r"^БИБЛИОГРАФИЧЕСКИЕ ССЫЛКИ\.?:?$|"
	r"^БИБЛИОГРАФИЧЕСКИЕ ССЫЛКИ и объектные ресурсы\.?:?$|"
	r"^БИБЛИОГРАФИЧЕСКИЙ СПИСОК\.?:?$|"
	r"^БИБЛИОГРАФИЯ\.?:?$|"
	r"^БИБЛИОГРАФИЯ издания\.?:?$|"
	r"^БИБЛИОГРАФИЧЕСКИЙ$|"
	r"^СПИСОК СОКРАЩЕНИЙ\.?:?$|"
	r"^СПИСОК литературы\.?:?$|"
	r"^СПИСОК рекомендуемой литературы\.?:?$|"
	r"^СПИСОК ИЛЛЮСТРАЦИЙ\.?:?$|"
	r"^СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ\.?:?$|"
	r"^СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ И ЛИТЕРАТУРЫ\.?:?$|"
	r"^СПИСОК использованной литературы\.?:?$|"
	r"^СПИСОК использованных в работе источников\.?:?$|"
	r"^СПИСОК ГРУППЫ\.?:?$|"
	r"^ПРИЛОЖЕНИЕ\.?[\t\s]*[IVXLCDM]*(no )?\d*\.?:?$",
	re.IGNORECASE
)
RE_AUTHORS = re.compile(r"^Кукаркин Евгений|")
RE_DIVIDER = re.compile(r"^(\.\s){3,}|^[-–—]{3,}|^-$")
RE_TITLE = re.compile(
	r'^(\d+|[IVXLCDM]+)?\.?\s?(глава|часть|книга|аннотация|введение|пролог|эпилог|рассказ|предисловие)(?![а-яёА-ЯЁ])[.\s]?(\d+|[IVXLCDM]+)?(.*)?["»]?$',
	re.IGNORECASE
)
RE_SINGLE_NUMBER = re.compile(r"^\d+$|^(\d+.)+$")
RE_TOC_LINE = re.compile(r"\.{3,}(\s*)?\d+(\s*)?(стр)?\.?$")  # Ловит примеры типа: "Глава 1. Начало пути ........ 14"
RE_BIBLIOGRAPHY = re.compile(
	r"^\d+\.?(\s+)?.*(?:"  # Начинается с цифры и пробела, а дальше:
	r"//|"  # Либо двойной слэш (главный маркер советской библиографии)
	r"\s[Сс]\.\s*\d+|"  # Либо "С. 44" / "с. 44" (страницы)
	r"\s[Pp]\.\s*\d+|"  # Либо "P. 44" / "p. 44" (английские страницы)
	r"\s[Тт]\.\s*(?:\d+|[IVXLCDM]+)|"  # Либо "Т. 29" / "Т. XXII" (тома)
	r"\sМ\.?[\s:]|"
	r"\d\d\d\d\.?$"  # Оканчивается на год
	r")"
)

# Косметика
RE_SPACES = re.compile(r"[\s\t]+")
RE_TRANSFER = re.compile(r"[а-яa-z]-$")
RE_START_DASH = re.compile(r"^[—–\-]{1,2}")
# RE_DASH = re.compile(r"(?<=.)[—–\-]{1,2}")
# RE_DASH = re.compile(r"[—–\-]{1,2}")
RE_DASH = re.compile(r"(?<!^)[—–\-]{1,2}")
RE_SEPARATE_DASH = re.compile(r"(?<!^)(?<=\s)-")
RE_SPECIAL_SYMBOLS = re.compile(r"[\\*_#„“‘’]")

# Валидные символы (Кириллица, базовая латиница, цифры и стандартная пунктуация)
RE_VALID_CHARS = re.compile(r"[а-яА-ЯёЁa-zA-Z0-9\s.,!?–—\-\"\'()«»:;№%=+*\x0c]")
RE_LATIN_ONLY = re.compile(r"[a-zA-Z]")


class LineType(Enum):
	UNKNOWN = auto()
	# SENTENCE_BEGIN = auto()  # Начало предложения
	# DIALOGUE_BEGIN = auto()  # Начало диалога
	# CONTINUATION = auto()  # Продолжение предложения
	# END = auto()  # Конец предложения
	TITLE = auto()  # Заголовок
	# LIST_ITEM = auto()  # Элемент списка
	BREAK = auto()  # Предложение оборвалось буквой, цифрой, чем бы то ни было
	DASH = auto()  # Предложение оборвалось тире
	HYPHEN = auto()  # Предложение оборвалось дефисом
	TRANSFER = auto()  # Предложение оборвалось переносом слова
	COMPLETED = auto()  # Завершённое предложение или параграф
	TRASH_SECTION = auto()  # Оглавление, список литературы и прочее
	BIBLIOGRAPHY_ITEM = auto()  # Элемент списка литературы
	TOC_ITEM = auto()  # Элемент оглавления
	DIVIDER = auto()  # Разделитель чего бы то ни было
	SINGLE_NUMBER = auto()  # Одиночный номер (год, страница)
	METADATA = auto()  # Метаданные


def _normalize_line(line: str) -> str:
	""" Очистка строки от не utf-8 символов """
	line = re.sub(r'[\u200b\u200e\u200f\u202a-\u202e]', '', line)
	line = unicodedata.normalize("NFKC", line)
	line = line.replace('\x0c', '').replace('\x1a', '').strip()
	if not line:
		return ""
	return line


def _filter_line(line: str, threshold: float = 0.85) -> str:
	"""
	Детекция загагулин (Грузинский, Санскрит и т.д.).
	Если доля валидных символов меньше, чем threshold,
	то возвращаем пустую строку.
	:param threshold: Минимальная доля валидных символов в строке.
	"""

	valid_count = len(RE_VALID_CHARS.findall(line))
	total_count = len(line)

	if total_count > 0 and total_count - valid_count > 4 and (valid_count / total_count) < threshold:
		return ""  # Если в строке >15% чужого алфавита (и более 4 символов ) или битой кодировки — сжигаем её целиком
	return line


def _clean_line(line: str) -> str:
	""" Базовая косметика текста """
	# Спецсимволы -> пустая строка
	line = RE_SPECIAL_SYMBOLS.sub("", line)
	# Множественные пробелы/табуляции -> пробел
	line = RE_SPACES.sub(" ", line)
	# Одна или две чёрточки в начале строки -> длинное тире (---) (начало диалога)
	line = RE_START_DASH.sub("—", line)
	# Одна или две чёрточки НЕ в начале строки -> минус.
	# (После строки кода ниже, в line НЕ ОСТАНЕТСЯ никаких чёрточек,
	# кроме минуса, за исключением первого символа
	# строки, если это диалог)
	line = RE_DASH.sub("-", line)
	# Минусы, не в начале строки, с пробелом перед ним -> длинное тире.
	# Таким образом дефисы и переносы слов сохраняются
	line = RE_SEPARATE_DASH.sub("—", line)
	line = re.sub(r"^==", "—", line)
	line = re.sub(r"(?<!^)(?<=\s)==", "—", line)

	return line.strip()


def _get_line_type(line: str, next_line: str = "") -> LineType:
	"""
	Возвращает тип целевой строки. Тут важен приоритет.
	Если строка является обычным предложением,
	при этом она же является метаданными, то
	мы объявляем её метаданными!
	"""

	if RE_META.search(line):
		return LineType.METADATA
	elif RE_SINGLE_NUMBER.match(line):
		return LineType.SINGLE_NUMBER
	elif RE_DIVIDER.match(line):
		return LineType.DIVIDER
	elif RE_TRASH_SECTIONS.match(line):
		return LineType.TRASH_SECTION
	elif RE_TOC_LINE.search(line):
		return LineType.TOC_ITEM
	elif RE_BIBLIOGRAPHY.search(line):
		return LineType.BIBLIOGRAPHY_ITEM
	# elif target_line.endswith(" –"):
	#     return LineType.DASH
	elif RE_COMPLETED.search(line):
		return LineType.COMPLETED

	# Если скрипт дошёл до сюда, то target_line - это незавершённое
	# предложение и без next_line мы не сможем определить его тип
	elif next_line:
		# next_line - начало диалога
		if next_line.startswith("—"):
			return LineType.TITLE
		# next_line - НЕ начало диалога
		else:
			if RE_TRANSFER.search(line) and next_line[0].islower():
				return LineType.TRANSFER
			elif RE_TRANSFER.search(line) and not next_line[0].islower():
				return LineType.HYPHEN
			elif line.endswith(" —"):  # тип DASH подразумевает, что после целевой строки есть еще одна строка
				return LineType.DASH
			elif RE_TITLE.match(line) or (not next_line[0].islower() and len(line) < 30):
				return LineType.TITLE
			else:
				return LineType.BREAK
	else:
		return LineType.UNKNOWN


def _process_document(
		raw_lines: list,
		not_valid_threshold: float = 0.85,
		english_threshold: float = 0.30,
		drop_trash_sections_threshold: float = 0.80
) -> str:
	"""
	Обрабатывает один целый документ между <startoftext> и <endoftext>.
	Самих <startoftext> и <endoftext> тут нет.
	"""
	cleaned_lines = []
	target_line = ""

	# Переменные для подсчета языка во всем документе
	total_chars = 0
	latin_chars = 0

	if not raw_lines:
		return ""

	for i, line in enumerate(raw_lines):
		line = line.strip()
		if not line:
			continue

		# Отсеиваем ненужные строки до их очистки
		line_type = _get_line_type(line)
		if line_type in (
				LineType.METADATA,
				LineType.SINGLE_NUMBER,  # TODO отсеивать после очистки?
				LineType.DIVIDER,
				LineType.TOC_ITEM,
				LineType.BIBLIOGRAPHY_ITEM  # TODO отсеивать после очистки?
		):
			continue

		if line_type == LineType.TRASH_SECTION:
			if i / len(raw_lines) > drop_trash_sections_threshold:
				break

		line = _normalize_line(line)
		line = _filter_line(line, threshold=not_valid_threshold)
		line = _clean_line(line)
		if not line:
			continue

		# Считаем статистику латиницы для детекции чисто английских текстов
		total_chars += len(line)
		latin_chars += len(RE_LATIN_ONLY.findall(line))

		if target_line:
			target_line_type = _get_line_type(target_line, line)
			if target_line_type in (LineType.COMPLETED, LineType.TITLE):
				cleaned_lines.append(target_line)
			elif target_line_type == LineType.TRANSFER:
				target_line = target_line[:-1] + line
				continue
			elif target_line_type == LineType.HYPHEN:
				target_line += line
				continue
			elif target_line_type in (LineType.DASH, LineType.BREAK):
				target_line = target_line + " " + line
				continue

		target_line = line

	if target_line:
		cleaned_lines.append(target_line)

	# Если в рамках ВСЕГО документа латиницы больше 30% — это английская статья/книга.
	# Выкидываем её целиком.
	if total_chars > 0 and (latin_chars / total_chars) > english_threshold:
		return ""

	return "\n".join(cleaned_lines).strip()


def process_corpus(
		input_path,
		output_path,
		not_valid_threshold: float = 0.85,
		english_threshold: float = 0.30
):
	start_time = time.time()
	total_size = os.path.getsize(input_path)
	processed_bytes = 0
	counter = 0

	seen_hashes = set()  # D5-хеши уникальных текстов
	current_doc_lines = []
	in_doc = False

	stats = {"total_docs": 0, "written_docs": 0, "duplicates": 0, "english_dropped": 0}

	print(f"Старт мега-очистки: {input_path}")

	with open(input_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
			open(output_path, 'w', encoding='utf-8') as f_out:

		for line in f_in:
			counter += 1
			processed_bytes += len(line.encode('utf-8', errors='ignore'))
			clean_l = line.strip()

			if clean_l == "<startoftext>":
				in_doc = True
				current_doc_lines = []
				continue

			if clean_l == "<endoftext>":
				in_doc = False
				stats["total_docs"] += 1

				# Обрабатываем накопленный документ
				doc_text = _process_document(current_doc_lines, not_valid_threshold, english_threshold)

				if not doc_text:
					# Документ отфильтрован (например, был чисто английским)
					stats["english_dropped"] += 1
					continue

				# === Детектор Дубликатов ===
				doc_hash = hashlib.md5(doc_text[:3000].encode('utf-8')).hexdigest()
				if doc_hash in seen_hashes:
					stats["duplicates"] += 1
					continue  # Скипаем дубликат текста

				seen_hashes.add(doc_hash)
				stats["written_docs"] += 1

				# Пишем чистый документ на диск
				f_out.write("<startoftext>\n" + doc_text + "\n<endoftext>\n")
				continue

			if in_doc:
				current_doc_lines.append(line)

			if processed_bytes and counter % 100_000 == 0:
				print(
					f"\rПрогресс: {(processed_bytes / total_size) * 100:.1f}% | Уникальных книг: {stats['written_docs']}",
					end="",
					flush=True
				)

	print(f"\n\n=== СТАТИСТИКА ОЧИСТКИ ===")
	print(f"Всего документов в сыром файле: {stats['total_docs']}")
	print(f"Удалено дубликатов: {stats['duplicates']}")
	print(f"Удалено английских/пустых текстов: {stats['english_dropped']}")
	print(f"Сохранено чистых уникальных документов: {stats['written_docs']}")
	print(f"Время работы: {time.time() - start_time:.2f} сек.")


def build_dataset(ds_dir, cleaned_dir):
	DS_FILES = [
		# f"{ds_dir}/merged_data_1.txt",
		f"{ds_dir}/merged_data_2.txt",
		# f"{ds_dir}/merged_data_3.txt",
		# f"{ds_dir}/merged_data_4.txt",
		# f"{ds_dir}/merged_data_5.txt",
		# f"{ds_dir}/merged_data_6.txt",
		# f"{ds_dir}/merged_data_7.txt",
	]

	OUT_FILES = [
# 		f"{cleaned_dir}/cleaned_data_1.txt",
		f"{cleaned_dir}/cleaned_data_2.txt",
		# f"{cleaned_dir}/cleaned_data_3.txt",
		# f"{cleaned_dir}/cleaned_data_4.txt",
		# f"{cleaned_dir}/cleaned_data_5.txt",
		# f"{cleaned_dir}/cleaned_data_6.txt",
		# f"{cleaned_dir}/cleaned_data_7.txt",
	]
	for ds_file, clean_file in zip(DS_FILES, OUT_FILES):
		process_corpus(ds_file, clean_file)


if __name__ == '__main__':
	build_dataset(DS_DIR, CLEANED_DIR)
	# for ds_file, clean_file in zip(DS_FILES, OUT_FILES):
	# process_corpus("test.txt", "test_cleaned.txt")
# process_corpus(DS_DIR, CLEANED_DIR)
