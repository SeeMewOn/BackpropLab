import os
import time
from enum import Enum

import unicodedata
from datasets import load_dataset
import re

DS_DIR = "../../../data/datasone_wiki"
CLEANED_DIR = "../../../data/CLEANED_DATASONE"
STATS_DIR = "../../../data/CLEANED_DATASONE/stats"


class ArticleType(Enum):
	""" Причины, по которым статья считается мусорной. """
	FUNDAMENTAL = "Хорошая статья для пре-трейна"
	# === Причины по названию (title) ===
	BIO = "Личности <Фамилия, Имя Отчество> или что-то имени чего-то"
	NAMED_AFTER = "имени, им."
	ROMAN_NUMS = "Римские цифры: <Гексанитрокобальтат(III) калия>, <Ричард III> и т.д."
	DATES = "Даты: <27 февраля>, <1654 год>"
	BAD_START_WORD = "Нежелательные слова в начале статьи: список, хронология и т.д."
	GEOGRAPHY = "Географические объекты"
	VALUES = "Название (значения)"
	SPORT = "Олимпийские игры, чемпионаты и тд."

	# === Причины по контенту (text) ===
	TOO_SHORT_RAW = "Статья изначально короче 3000 симв."
	TOO_SHORT_POST_CLEAN = "Статья стала короче 2000 симв. после обрезки хвостов"
	# Конкретные маркеры мусора в теле статьи
	GEOPOLITICS = "Геополитический маркер (государство, суверенный...)"
	BIO_CARD = "Биографическая карточка (писатель, царь, даты жизни...)"
	POLITICS_MILITARY = "Политики, чиновники, военный шум"
	INFRASTRUCTURE = "Статичные объекты (собор, музей, завод...)"
	GEOGRAPHY_CONTENT = "Мелкий мусор (село, река, переменная звезда...)"
	ASTRONOMY = "Звёзды, созвездия"


# -------------------------------------------------------------------------------------------------------- #
#                                        Фильтрация названий статей                                        #
# -------------------------------------------------------------------------------------------------------- #


# Все биографии людей в Википедии называются по схеме Фамилия, Имя Отчество.
# Например: «Бродский, Иосиф Александрович», «Меир, Голда».
# Если в названии есть запятая, а после неё пробел и Большая буква — это 100% человек.
RE_BIO = re.compile(r'^[А-ЯЁа-яё\-\'’]+\s*,\s+[А-ЯЁ]')  # "Фамилия, Имя" или что-то имени кого-то
# RE_BIO = re.compile(
#     r'^[А-ЯЁ][а-яё\-’\']+'          # Первая часть фамилии (с заглавной, буквы/дефисы/апострофы)
#     r'(?:\s+[А-ЯЁ][а-яё\-’\']+){0,2}' # Возможные 2-е и 3-е слова фамилии (напр. "Сантус" или "Борда")
#     r'\s*,\s*'                        # Запятая с возможными пробелами вокруг
#     r'[А-ЯЁ][а-яё\-’\']+'          # Имя (ОБЯЗАТЕЛЬНО с заглавной буквы)
#     r'(?:\s+[А-ЯЁ][а-яё\-’\']+){0,2}' # Возможные Отчество / второе имя (с заглавной)
#     r'(?:\s*\(.+\))?$'                # Необязательное уточнение в скобках на конце, напр. "(физик)"
# )
RE_NAMED_AFTER = re.compile(r"\s+им\.|\sимени\s", re.IGNORECASE)
RE_ROMAN_NUMS = re.compile(r'\b(?!(?:ML|DL)\b)[IVXLCDM]+\b')  # Наличие римских цифр (Петр I, Людовик XIV) (Кроме ML/DL)
RE_DATES = re.compile(r"\b\d+\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)|"
                      r"(\b\d+(-й|-го|-е)?\s+год)", re.IGNORECASE)
RE_BAD_START_WORDS = re.compile(  # Нежелательные стартовые слова
	r"^(список|премия|хронология|история|география|население|экономика|культура|политика|герб|"
	r"флаг|гимн|календарь|словарь|атлас|фильмография|улица|район|районы|единая россия)\s+",
	re.IGNORECASE)
RE_GEOGRAPHY = re.compile(
	r"я область|сельсовет|сельские советы|поселение|департамент|улица|переулок|"
	r"памятник|\(станция\)|\(станция метро\)|\(округ\)|административный округ|автономный округ|"
	r"\(аэропорт\)|\(регион\)|\(провинция\)|\(коммуна\)|\(остров\)|(?<![А-Яа-яёЁ])парк(?![а-яёА-ЯЁ])",
	# Никольск (Вологодская область), Горно-Бадахшанская автономная область
	re.IGNORECASE
)
RE_VALUES = re.compile(r"\(значения\)|\(фамилия\)|^\.")  # Список значений слова, фамилии, домены
RE_SPORT = re.compile(r"чемпион|олимпийск|футбол|хоккей|кёрлинг|волейбол|теннис", re.IGNORECASE)
RE_MILITARY = re.compile(r"батальон|(?<![А-Яа-яёЁ])полк(?![а-яёА-ЯЁ])|дивизия|армия|(?<![А-Яа-яёЁ])крейсер(?![а-яёА-ЯЁ])", re.IGNORECASE)
RE_POLITICS = re.compile(r"выборы", re.IGNORECASE)

TITLE_RULES_MAPPING = {
	ArticleType.BIO: RE_BIO,
	ArticleType.NAMED_AFTER: RE_NAMED_AFTER,
	ArticleType.ROMAN_NUMS: RE_ROMAN_NUMS,
	ArticleType.DATES: RE_DATES,
	ArticleType.BAD_START_WORD: RE_BAD_START_WORDS,
	ArticleType.GEOGRAPHY: RE_GEOGRAPHY,
	ArticleType.VALUES: RE_VALUES,
	ArticleType.SPORT: RE_SPORT,
	ArticleType.POLITICS_MILITARY: RE_MILITARY,
	ArticleType.GEOPOLITICS: RE_POLITICS,
}
TITLE_RULES = [(pattern, reason) for reason, pattern in TITLE_RULES_MAPPING.items()]
# TITLE_RULES = [
# 	(RE_BIO, ArticleType.BIO),
# 	(RE_ROMAN_NUMS, ArticleType.ROMAN_NUMS),
# 	(RE_DATES, ArticleType.DATES),
# 	(RE_BAD_START_WORDS, ArticleType.BAD_START_WORD),
# 	(RE_GEOGRAPHY, ArticleType.GEOGRAPHY),
# 	(RE_VALUES, ArticleType.VALUES),
# ]

# -------------------------------------------------------------------------------------------------------- #
#                                        Фильтрация содержания статей                                      #
# -------------------------------------------------------------------------------------------------------- #


TEXT_RULES_MAPPING = {
	ArticleType.GEOPOLITICS: [
		# 1. Страны, материки, геополитика и территории
		r"[—–\-] (государство|островное государство|суверенное государство|королевство|"
		r"эмират|материк|континент|административный центр|муниципальное образование|муниципальный район)(?![а-яёА-ЯЁ])",
		r"официальное название [—–\-] (Респу́блика|Республика|Короле́вство|Королевство|Госуда́рство|Государство)",
		r"в (Юго-Восточной|Западной|Центральной|Южной|Восточной|Северной|Евроазиатской) (Африке|Азии|Америке|Европе)",
		r"омывается\s+[а-яёА-ЯЁ]+\s+морем",
		r"на (зимних|летних)?\s*Олимпийских играх",
	],
	ArticleType.BIO_CARD: [
		# 2. Личности, правители, философы, писатели (Выжигаем био-карточки)
		# Используем \b(философ)\b, чтобы "философия" или "философский" НЕ триггерились
		r"[—–\-] ([а-яёА-ЯЁ\s,-]+)?(?<![А-Яа-яёЁ])(философ|врач|писатель|поэт|историк|учёный|ученый|мыслитель|космолог|мифограф"
		r"|исследователь|художник|художница|архиерей|епископ|живописец|архитектор|музыкант|актёр|актер|актриса|певец|"
		r"певица|дирижёр|гравёр|спортсмен|футболист|хоккеист|теннисист|боксёр|шахматист"
		r"скульптор|композитор|режиссёр|режиссер)(?![А-Яа-яёЁ])",
		r"[—–\-] ([а-яёА-ЯЁ\s,-]+)?(?<![А-Яа-яёЁ])(царь|король|император|императрица|принцесса|герцог|герцогиня|князь|княгиня|магистр|султан|хан|шах)(?![А-Яа-яёЁ])",
		r"из династии (Романовых|Рюриковичей|Бурбонов|Габсбургов|династии)",
		r"\((ок\.\s*)?\d+\s+(до н\. э\.|н\. э\.)?[\s\S]*?[—–\-]\s*(ок\.\s*)?\d+",
		# Даты жизни: (ок. 315 до н. э. — ок. 240 до н. э.)
		r"(?<![А-Яа-яёЁ])(род\.|родился|родилась|умер|умерла)\s+\d+",  # Даты рождения и смерти
	],
	ArticleType.POLITICS_MILITARY: [
		# 3. Политики, чиновники, военные (Весь ура-патриотический и госслужебный шум)
		r"[—–\-] ([а-яёА-ЯЁ\s,-]+)?(?<![А-Яа-яёЁ])(полководец|генерал|адмирал|маршал|полковник|командир|офицер|военачальник|боец|бригадир)(?![А-Яа-яёЁ])",
		r"[—–\-] ([а-яёА-ЯЁ\s,-]+)?(?<![А-Яа-яёЁ])(президент|премьер-министр|министр|депутат|революционер|политик|государственный деятель|дипломат|предприниматель|бизнесмен)(?![А-Яа-яёЁ])",
		r"[—–\-] (воинское соединение|дивизия|полк|бригада|истребительный авиационный полк|эскадрилья)(?![А-Яа-яёЁ])",
		r"(руководителей СССР|первых лиц|коммунистической партии|выборы президента)",
	],
	ArticleType.INFRASTRUCTURE: [
		# 4. Постройки, музеи, соборы, замки (Статичные объекты инфраструктуры)
		r"[—–\-] ([а-яёА-ЯЁ\s,-]+)?(?<![А-Яа-яёЁ])(собор|храм|замок|дворец|музей|кабинет|камора|памятник архитектуры|монастырь|церковь|синагога|разрез|предприятие|завод|фабрика)(?![А-Яа-яёЁ])",
		r"(?<![А-Яа-яёЁ])(кафедральный собор|музей|шедевр искусства)(?![А-Яа-яёЁ])",
	],
	ArticleType.GEOGRAPHY_CONTENT: [
		# 5. Базовый мусор (села, реки, переменные звезды из каталогов)
		r"[—–\-] (село|деревня|посёлок|поселок|коммуна|река|протока|город|озеро|остров|мыс|округ"
		r"|уезд|район|альбом|сингл|саундтрек|композиция|трек|персонаж)(?![А-Яа-яёЁ])",

	],
	ArticleType.ASTRONOMY: [
		r"[—–\-] (одиночная|переменная|двойная)?\s*звезда в созвездии"
	],
}

# Компилируем один раз при старте модуля
TEXT_RULES = [
	(re.compile(pattern, re.IGNORECASE), reason)
	for reason, patterns in TEXT_RULES_MAPPING.items()
	for pattern in patterns
]

# -------------------------------------------------------------------------------------------------------- #
#                                        Удаление ненужных разделов                                        #
# -------------------------------------------------------------------------------------------------------- #


TRASH_SECTION_PATTERNS = [
	# 1. COMMON
	r"^См\. также\s*$", r"^Примечания\s*$", r"^Ссылки\s*$", r"^Литература\s*$", r"^Источники\s*$",
	r"^В ролях\s*$", r"^Создатели\s*$", r"^Съёмочная группа\s*$",
	# 2. Фильмы
	r"^Награды\s*$", r"^Премии\s*$", r"^Номинации\s*$", r"^Награды и номинации\s*$",
	r"^Критика\s*$", r"^Саундтрек\s*$", r"^Кассовые сборы\s*$",
	r"^Галерея\s*$", r"^Демография\s*$"
]

# Собираем всё в одну мега-регулярку для дикой скорости
# Флаг re.MULTILINE заставляет ^ триггериться на новую строку внутри текста
RE_TRUNCATE_SECTIONS = re.compile("|".join(TRASH_SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


# -------------------------------------------------------------------------------------------------------- #
#                                      Создание чистого датасета                                           #
# -------------------------------------------------------------------------------------------------------- #

def _is_fundamental(
		title: str,
		text: str = "",
		min_article_length: int = 3000,
		check_first_k: int = 500,
) -> tuple[bool, ArticleType]:
	"""
	Фундаментальная ли статья?
		- title is not None, text = None -> Проверка фундаментальности названия статьи.
		- title = None, text is not None -> Проверка фундаментальности текста статьи.
		- title & text is not None -> Проверка на фундаментальность названия и текста статьи вместе.
	:param min_article_length: Минимальная длина статьи, которая пройдёт проверку.
	Если 0, то ограничений по размеру статьи снизу нет.
	:param check_first_k: Сколько первых символов статьи проверять.
	:return: Фундаментальная ли статья и тип мусора. Если статья фундаментальна - тип мусора - None
	"""
	# Проверка названия
	if title:
		for pattern, reason in TITLE_RULES:
			if pattern.search(title):
				return False, reason
		# if RE_BIO.match(title): return False, ArticleType.BIO
		# if RE_ROMAN_NUMS.search(title): return False, ArticleType.ROMAN_NUMS
		# if RE_DATES.search(title): return False, ArticleType.DATES
		# if RE_BAD_START_WORDS.match(title): return False, ArticleType.BAD_START_WORD
		# if RE_GEOGRAPHY.search(title): return False, ArticleType.GEOGRAPHY
		# if RE_VALUES.search(title): return False, ArticleType.VALUES

	# Проверка длины статьи
	if text:
		if min_article_length > 0:
			if len(text) < min_article_length:
				return False, ArticleType.TOO_SHORT_RAW

		# Проверяем первые check_first_k символов. Там всегда сидит определение статьи
		definition_zone = text[:check_first_k]

		for pattern, reason in TEXT_RULES:
			if pattern.search(definition_zone):
				return False, reason  # Нашли мусорный маркер -> выкидываем статью

	return True, ArticleType.FUNDAMENTAL  # Статья прошла все круги ада, она чистая


def _truncate_sections(text: str) -> str:
	if not text:
		return ""

	# Ищем первое вхождение любого мусорного заголовка, стоящего на новой строке
	match = RE_TRUNCATE_SECTIONS.search(text)

	if match:
		# Отрезаем всё, что идет начиная с индекса старта этого заголовка
		return text[:match.start()].strip()

	return text.strip()


def _normalize_text(text: str):
	text = re.sub(r'[\u200b\u200e\u200f\u202a-\u202e]', '', text)
	text = unicodedata.normalize('NFD', text)  # Разбиваем все составные буквы (и латинские é, и кириллические а́)
	text = text.replace('\u0301', '')  # Вырезаем само ударение
	text = unicodedata.normalize('NFKC', text)  # Возвращаемся обратно
	text = re.sub(r'\s*\(\s*\)', '', text)  # Удаление пустых скобок
	text = re.sub(r'\[.*?]', '', text)  # Удаление квадратных скобок и всего, что внутри них
	text = re.sub(r"[—–]", "-", text)  # Все чёрточки превращаются в минусы
	text = re.sub(r'[ \t]+', ' ', text)  # Схлопываем множественные пробелы в один
	text = re.sub(r'\n\s*\n+', '\n', text)  # Удаление множественных переносов строк
	return text.strip()


def _inspect_filtered_dataset(dataset, interval: tuple):
	print("Загрузка датасета...")

	print(f"Всего статей в сыром дампе: {len(dataset)}")
	print("Начинаем стриминг и фильтрацию. Выводим качественные статьи:\n" + "=" * 60)

	printed_count = 0
	scanned_count = 0

	# Идем по датасету линейно
	for item in dataset.select(range(interval[0], interval[1])):
		scanned_count += 1
		text = item['text']
		title = item['title']

		if _is_fundamental(title, text):
			printed_count += 1
			# print(f" СЕМПЛ #{scanned_count} (Длина: {len(text)} симв.)")
			# Если есть колонка title, выведем её

			print(f"|{title}| #{scanned_count} (Длина: {len(text)} симв.)")

			# print("-" * 40)
			# Принтим первые 1200 символов статьи для оценки глубины контента
			print(text)
			print("\n")

	print(f"Сканирование завершено. Проверено статей: {scanned_count}. Найдено годноты: {printed_count}")


def _get_article_titles(
		dataset,
		fundamental_path,
		garbage_path,
		min_article_length=500,
		check_first_k=300,
		size_step=1000,
):
	""" Генерация файла со всеми названиями статей датасета """
	print(f"Всего статей в сыром дампе: {len(dataset)}")

	stats = {
		"total_articles": len(dataset),
		"FUNDAMENTAL": 0,
		"BIO": 0,
		"NAMED_AFTER":0,
		"ROMAN_NUMS": 0,
		"DATES": 0,
		"BAD_START_WORD": 0,
		"GEOGRAPHY": 0,
		"VALUES": 0,
		"SPORT": 0,
		"TOO_SHORT_RAW": 0,
		"TOO_SHORT_POST_CLEAN": 0,
		"GEOPOLITICS": 0,
		"BIO_CARD": 0,
		"POLITICS_MILITARY": 0,
		"INFRASTRUCTURE": 0,
		"GEOGRAPHY_CONTENT": 0,
		"ASTRONOMY": 0,
	}
	sizes = {}

	def save_size(text_len):
		# Вычисляем нужный диапазон (округляем вверх до ближайшего шага)
		k = ((text_len + size_step - 1) // size_step) * size_step

		if k in sizes:
			sizes[k] += 1
		else:
			sizes[k] = 1

	with open(fundamental_path, "w", encoding="utf-8") as f_fun, open(garbage_path, "w", encoding="utf-8") as f_garb:
		start_time = time.time()
		progress = 0
		for item in dataset:
			title = item['title']
			text = item['text']
			is_fundamental, trash_type = _is_fundamental(
				title=title,
				text=text,
				min_article_length=min_article_length,
				check_first_k=check_first_k,
			)
			if is_fundamental:
				f_fun.write(f"{title} - [{len(text)}]" + "\n")
				stats["FUNDAMENTAL"] += 1
				save_size(len(text))
			else:
				f_garb.write(f"{title} - {trash_type.name} - [{len(text)}]" + "\n")
				stats[trash_type.name] += 1

			progress += 1
			if progress % 100_000 == 0:
				t = time.time() - start_time
				v = progress / t
				print(f"\rProgress: {progress / len(dataset) * 100:.2f}% | "
				      f"Time: {t:.2f} s | "
				      f"Time Left: {(len(dataset) - progress) / v:.2f} s | "
				      f"Speed: {v:.2f} Articles / s | {stats}", end="", flush=True)
		print()
		print(f"Time: {time.time() - start_time:.2f}s")

		print(f"Всего фундаментальных названий: {stats["FUNDAMENTAL"]}")
		print(stats)
		print(sizes)


def print_articles_by_titles(dataset, titles: list[str], first_k=500, clean: bool = False):
	for item in dataset:
		if item["title"] in titles:
			title = item["title"]
			text = item["text"][:first_k]
			if clean:
				text = _truncate_sections(text)
				text = _normalize_text(text)
			print(f"============================================ {title} ============================================")
			print(text)
			print()


def build_dataset(
		dataset,
		output_dir,
		start_tag  = "<startoftext>",
		end_tag = "<endoftext>",
		max_shard_size_gb=1.0,
		check_first_k=500,
		min_dirty_article_length=3000,
		min_clean_article_length=2500,
):
	"""Пайплайн фильтрации, очистки и нарезки датасета на txt шарды"""
	os.makedirs(output_dir, exist_ok=True)

	max_bytes = max_shard_size_gb * 1024 * 1024 * 1024
	shard_index = 1
	current_shard_bytes = 0
	start_time = time.time()
	processed_count = 0
	saved_count = 0

	# Открываем первый шард
	shard_path = os.path.join(output_dir, f"wiki_shard_{shard_index}.txt")
	f = open(shard_path, "w", encoding="utf-8")
	f_titles_fund = open(f"{STATS_DIR}/titles_fundamental.txt", "w", encoding="utf-8")
	f_titles_garb = open(f"{STATS_DIR}/titles_garbage.txt", "w", encoding="utf-8")
	f_stats = open(f"{STATS_DIR}/stats.txt", "w", encoding="utf-8")

	stats = {
		"total_articles": len(dataset),
		"FUNDAMENTAL": 0,
		"BIO": 0,
		"ROMAN_NUMS": 0,
		"DATES": 0,
		"BAD_START_WORD": 0,
		"GEOGRAPHY": 0,
		"VALUES": 0,
		"SPORT": 0,
		"TOO_SHORT_RAW": 0,
		"TOO_SHORT_POST_CLEAN": 0,
		"GEOPOLITICS": 0,
		"BIO_CARD": 0,
		"POLITICS_MILITARY": 0,
		"INFRASTRUCTURE": 0,
		"GEOGRAPHY_CONTENT": 0,
		"ASTRONOMY": 0,
	}


	# Формируем шарды
	print(f"Запуск пайплайна. Целевой размер шарда: {max_shard_size_gb} ГБ")
	for item in dataset:
		processed_count += 1
		title = item["title"]
		text = item["text"]

		# Проверка на фундаментальность
		if not _is_fundamental(
				title=title,
				text=text,
				min_article_length=min_dirty_article_length,
				check_first_k=check_first_k):
			continue

		# Отрезаем мусорные разделы и нормализуем текст
		text = _truncate_sections(text)
		text = _normalize_text(text)

		# На всякий случай проверяем длину после обрезки хвостов
		if len(text) < min_clean_article_length:
			continue

		# Формируем финальный блок для претрейна
		block = f"{start_tag}\n{title}\n{text}\n{end_tag}\n\n"
		block_bytes = len(block.encode('utf-8'))

		# Проверяем, не переполнился ли текущий файл
		if current_shard_bytes + block_bytes > max_bytes:
			f.flush()
			f.close()
			print(f" Сформирован шард {shard_index}: {current_shard_bytes / (1024 ** 2):.2f} МБ. Успешно записан.")

			# Открываем новый файл
			shard_index += 1
			shard_path = os.path.join(output_dir, f"wiki_shard_{shard_index}.txt")
			f = open(shard_path, "w", encoding="utf-8")
			current_shard_bytes = 0

		# Пишем блок в текущий шард
		f.write(block)
		current_shard_bytes += block_bytes
		saved_count += 1

		if processed_count % 100_000 == 0:
			elapsed = time.time() - start_time
			print(
				f"Обработано: {processed_count} | Сохранено годноты: {saved_count} | Прошло времени: {elapsed:.1f} сек")

	# Не забываем закрыть последний файл
	f.flush()
	f.close()

	print("\n" + "=" * 60)
	print(f"Пайплайн успешно завершен за {time.time() - start_time:.1f} сек!")
	print(f"Всего проверено статей: {processed_count}")
	print(f"Всего сохранено в чистый датасет: {saved_count}")
	print(f"Итоговое количество шардов: {shard_index}")


if __name__ == "__main__":
	# Запускаем просмотр (выведет первые 10 статей, прошедших фильтр)
	# inspect_filtered_dataset(HF_DATASET_PATH, max_to_print=1000)

	ds = load_dataset(
		"arrow",
		data_files=os.path.join(DS_DIR, "wikipedia_ru-train-*.arrow"),
		split="train"
	)

	_get_article_titles(
		ds,
		"fundamental_wiki_article_titles.txt",
		"garbage_wiki_article_titles.txt",
		min_article_length=300,
		check_first_k=1000,
	)

	# print_articles_by_titles(
	# 	ds,
	# 	[
	# 		"Советско-германские соглашения (1939—1941)",
	# 		"Великая Отечественная война",
	# 		"Украинцы в США",
	# 		"Ногай давысы",
	# 		"Венгрия на зимних Олимпийских играх 2010",
	# 		"Украинское сельское поселение",
	# 		"9 Июля (Санта-Фе)",
	# 		"Ваханский хребет",
	# 		"Парламентские выборы в Латвии(2006)",
	# 		"Старый Крым (санаторий)",
	# 		"Американская чёрная кряква",
	# 		"Франческо Сальвиати",
	# 		"Арвальские братья",
	# 		"147-я стрелковая дивизия (2-го формирования)",
	# 		"Площадь Тургенева",
	# 		"Мясная улица",
	# 		"Агасси(тауншип, Миннесота)",
	# 		"Агдер(тауншип, Миннесота)",
	# 	],
	# 	clean=False,
	# 	first_k=300
	# )
	# print_articles_by_titles(
	# 	ds,
	# 	[
	# 		".rw",
	# 		".sa",
	# 		".sc",
	# 		".sd",
	# 		".se",
	# 		".sg",
	# 		".sh",
	# 		".si",
	# 		".sj",
	# 		".sk",
	# 		".sl",
	# 		".sm",
	# 		".sn",
	# 		".so",
	# 		".sr",
	# 		".st",
	# 		".sv",
	# 	],
	# 	clean=True,
	# 	first_k=1_000_000
	# )
